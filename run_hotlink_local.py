import json
import pathlib
import time

from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache

import pandas
import psycopg

from hotlink import support_functions

import config
import hotlink_local


########## CONSTANTS #########
LOCATIONS = ['Redoubt', 'Spurr', 'Veniaminof', 'Semisopochnoi']

# dict to map the output column name to database variable name
VARIABLE_ID_MAP = {
    "Max Probability": 3,
    "Number Hotspot Pixels": 4,
    "Hotspot Radiative Power (W)": 5,
    "MIR Background Brightness Temperature": 8,
    "MIR Hotspot Brightness Temperature": 9,
    "MIR Hotspot Max Brightness Temperature": 10,
    "TIR Hotspot Brightness Temperature": 11,
    "TIR Hotspot Max Brightness Temperature": 12,
    "Solar Zenith": 6,
    "Solar Azimuth": 7,
    "Day/Night Flag":24,
    "Metadata": 13,
    # Metadata needs special handling (see below)
}

DEVICE_ID_MAP = {
    'viirs': 1,
    'modis': 2,
}

#############################


@contextmanager
def db_cursor(host, user, password, dbname=config.db_name, port=5432, autocommit=False):
    conn = psycopg.connect(host=host, user=user, password=password, dbname=dbname, port=port)
    cursor = conn.cursor()
    try:
        yield cursor
        if autocommit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        # Always rollback. If autocommit=True, then the
        # transaction will have already been commited, so this is "failsafe"
        conn.rollback()
        cursor.close()
        conn.close()

def preevents_cursor(readonly=True, autocommit=False):
    """
    Simple wrapper for the db_cursor context manager, defaulting all values
    with a simple flag to switch between read-only and read-write user.
    """
    user = config.db_read_user if readonly else config.db_write_user
    password = config.db_read_pass if readonly else config.db_write_pass
    return db_cursor(config.db_host, user, password, autocommit=autocommit)

@lru_cache(maxsize=None)
def load_volcs():
    # Load volcanoes from the PREEVENTS database
    with preevents_cursor() as cursor:
        cursor.execute("""
        SELECT
            longitude as lon,
            latitude as lat,
            volcano_name as name,
            elevation as elev,
            volcano_id as id
        FROM volcano
        WHERE observatory='avo'
        """)

        columns = [desc.name for desc in cursor.description]
        data = pandas.DataFrame(cursor.fetchall(), columns=columns)

    return data

def get_datastream_mapping(location):
    query = """
        SELECT
            datastreams.datastream_id,
            variables.variable_id,
            devices.device_id
        FROM datastreams
        INNER JOIN volcano ON datastreams.volcano_id = volcano.volcano_id
        LEFT JOIN variables ON variables.variable_id = datastreams.variable_id
        LEFT JOIN devices ON devices.device_id = datastreams.device_id
        WHERE volcano_name = %s
    """
    with preevents_cursor() as cursor:
        cursor.execute(query, (location,))
        rows = cursor.fetchall()

    # Invert the mapping: DB variable_name -> result key
    IV_MAP = {v: k for k, v in VARIABLE_ID_MAP.items()}

    mapping = {
        (IV_MAP.get(row[1], row[1]),
         row[2]
        ): row[0]
        for row in rows
    }
    return mapping


def get_volc(vent):
    VOLCS = load_volcs()
    if isinstance(vent, str):
        volc = VOLCS[VOLCS['name'].str.lower()==vent.lower()]
        if len(volc) == 0:
            raise ValueError("Specified volcano not found!")
    else:
        dists = support_functions.haversine_np(vent[1], vent[0], VOLCS['lon'], VOLCS['lat'])
        volc = VOLCS[dists==dists.min()]

    return volc.iloc[0]


def save_results(results, mapping):
    # Save results to PREEVENTS database
    results['Day/Night Flag'] = (results['Day/Night Flag'] == 'D').astype(int)
    with preevents_cursor(readonly=False) as cursor:
        for _, row in results.iterrows():
            sensor = row["Sensor"].lower() # Pull from results for good measure
            sensor = DEVICE_ID_MAP[sensor]
            timestamp = row["Date"]

            # Save the metadata record
            metadata = {"satellite": row["Satellite"], "sensor": row["Sensor"]}
            metadata_json = json.dumps(metadata)
            metadata_datastream = mapping.get(("Metadata", sensor))
            if metadata_datastream:
                cursor.execute(
                    """INSERT INTO datavalues (
                        datastream_id, timestamp, categoryvalue, last_updated
                    )
                    VALUES (%s, %s, %s, now())
                    ON CONFLICT (datastream_id, timestamp)
                    DO NOTHING
                    --UPDATE SET categoryvalue=EXCLUDED.categoryvalue, last_updated=now()
                    """,
                    (metadata_datastream, timestamp, metadata_json)
                )
            # Save the value records
            for result_key in VARIABLE_ID_MAP.keys():
                if result_key in row and not pandas.isna(row[result_key]):
                    key = (result_key, sensor)
                    datastream_id = mapping.get(key)
                    if datastream_id:
                        cursor.execute(
                            """
                            INSERT INTO datavalues (datastream_id, timestamp, datavalue, last_updated)
                            VALUES (%s, %s, %s, now())
                            ON CONFLICT (datastream_id, timestamp)
                            DO NOTHING
                            """,
                            (datastream_id, timestamp, float(row[result_key]))
                        )
                    else:
                        print(f"Warning: No datastream_id for {result_key} with sensor {sensor}")
        cursor.connection.commit()

def load_file_list() -> list[str| None, list[list[pathlib.PosixPath]]]:
    # Get a list of files
    input_path = pathlib.Path(config.DATA_PATH)
    df = None
    sat = None
    
    if not input_path.exists():
        return sat, [] # No input directory, no input files
    
    file_types = defaultdict(list)
    files = list(input_path.glob('*.h5'))
    for file in files:
        ftype = file.name.split('_')[0]
        file_types[ftype].append(file)
        
    viirs_keys = {'SVI04', 'SVI05', 'GITCO'}
    if viirs_keys.issubset(file_types):
        sat = 'viirs'
        df = hotlink_local.match_viirs(
            file_types['SVI04'],
            file_types['SVI05'],
            file_types['GITCO']
        )
  
    unexpected = set(file_types) - viirs_keys
    if unexpected:
        print(f"Warning: Found unexpected file types: {unexpected}")
        # sat = 'modis'
        # df = pandas.DataFrame({'file_1': files,})
        
    if df is None or df.empty:
        return sat, []
    
    # Extract the final list of files
    results = df.to_numpy().tolist()
    return sat, results
    
def main():
    print("Beginning processing")
    sat, files = load_file_list()
    print(f"Found {len(files)} file(s) of type {sat} to process")
    for file_list in files:
        print(f"Processing {file_list[0].name}")
        
        for loc in LOCATIONS:
            t1 = time.time()
    
            # Make sure we are using the canonical volcano.
            volc = get_volc(loc)
            elev = volc['elev']
            volc_name = volc['name']
    
            datastream_mapping = get_datastream_mapping(volc_name)
            try:
                results, meta = hotlink_local.get_results(loc, elev, file_list, sat)
            except hotlink_local.CoverageError as e:
                print(f"Insufficient coverage for volcano {volc_name}, {sat}. {e}")
                continue
    
            if not results.empty and meta['Result Count'] > 0:
                pass
                # save_results(results, datastream_mapping)
            else:
                print(f"No results to save for {volc_name} {sat}")
    
            print(f"Ran HotLINK for {loc} in {time.time() - t1} seconds")
            
        print(f"All volcs processed for file {file_list[0].name} Removing source files")
        for file in file_list:
            file.unlink()
            
    print("All files processed.")

if __name__ == "__main__":
    main()
