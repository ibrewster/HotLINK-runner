import logging
logging.basicConfig(
    level=logging.INFO,    # Set the minimum level to capture (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the structure
    datefmt='%Y-%m-%d %H:%M:%S'  # Optional: simplies the timestamp format
)

import json
import pathlib
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from functools import lru_cache

import pandas
import psycopg
import redis

from hotlink import support_functions

import config
import hotlink_local


########## CONSTANTS #########
LOCATIONS = [
    'Korovin',
    'Martin',
    'Amukta',
    'Chiginagak',
    'Seguam',
    'Kiska',
    'Kanaga',
    'Westdahl',
    'Akutan',
    'Kasatochi',
    'Makushin',
    'Gareloi',
    'Okmok',
    'Bogoslof',
    'Augustine', 
    'Pavlof',
    'Cleveland',
    'Shishaldin',
    'Great Sitkin',
    'Redoubt',
    'Spurr',
    'Veniaminof',
    'Semisopochnoi'
]

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

@lru_cache(None)
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


@lru_cache(None)
def get_volc(vent):
    VOLCS = load_volcs()
    if isinstance(vent, str):
        volc = VOLCS[VOLCS['name'].str.lower()==vent.lower()]
        if len(volc) == 0:
            raise ValueError(f"Specified volcano ({vent}) not found!. Canidates:\n{sorted(VOLCS['name'])}")
    else:
        dists = support_functions.haversine_np(vent[1], vent[0], VOLCS['lon'], VOLCS['lat'])
        volc = VOLCS[dists==dists.min()]

    return volc.iloc[0]

def get_start(datastreams):
    # Group datastream_ids by sensor
    viirs = DEVICE_ID_MAP['viirs']
    modis = DEVICE_ID_MAP['modis']
    sensor_ids = {viirs: [], modis: []}
    for (result_key, sensor), datastream_id in datastreams.items():
        if sensor in sensor_ids:
            sensor_ids[sensor].append(datastream_id)

    latest_timestamps = {}

    # Timestamps in the database are based on the filename, which is in
    # turn based on pass start. Searches, on the other hand, are based on
    # pass *end*, which is somewhat later. Hopefully 15 minutes is sufficient,
    # but we may need to adjust.
    QUERY = """
    SELECT COALESCE(
        MAX(timestamp)+'15 minute'::interval,
        now() - '7 days'::interval
    ) as latest_timestamp
    FROM datavalues
    WHERE datastream_id = ANY(%s)
    AND timestamp >= now() - '7 days'::interval
"""
    with preevents_cursor() as cursor:
        for sensor, ids in sensor_ids.items():
            if ids: # Should always be true
                cursor.execute(QUERY, (ids,))
                result = cursor.fetchone()
                latest_timestamps[sensor] = result[0]

    return latest_timestamps

def save_results(results, mapping):
    # Save results to PREEVENTS database
    results['Day/Night Flag'] = results['Day/Night Flag'].apply(lambda x: json.dumps({"day_night": x}))
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
                        if result_key == 'Day/Night Flag':
                            datafield = "categoryvalue"
                            datavalue = row[result_key]
                        else:
                            datafield = "datavalue"
                            datavalue = float(row[result_key])

                        sql = psycopg.sql.SQL("""
                            INSERT INTO datavalues (datastream_id, timestamp, {datafield}, last_updated)
                            VALUES (%s, %s, %s, now())
                            ON CONFLICT (datastream_id, timestamp)
                            DO NOTHING
                            """).format(
                                   datafield = psycopg.sql.Identifier(datafield)
                            )
                        cursor.execute(
                            sql,
                            (datastream_id, timestamp, datavalue)
                        )
                    else:
                        logging.warning(f"Warning: No datastream_id for {result_key} with sensor {sensor}")
        cursor.connection.commit()


def load_file_list() -> list[str| None, list[list[pathlib.Path]]]:
    # Get a list of files
    input_path = pathlib.Path(config.DATA_PATH)
    df = None
    sat = None

    if not input_path.exists():
        return sat, [] # No input directory, no input files

    files = pandas.DataFrame(
        [
            (f, f.name.split('_')[0], file_key(f))
            for f in input_path.rglob('*.h5')
        ],
        columns = ['path', 'type', 'key']
    )

    viirs_keys = {'SVI04', 'SVI05', 'GITCO'}
    if viirs_keys.issubset(files['type'].tolist()):
        sat = 'viirs'
        df = hotlink_local.match_viirs(
            files[files['type']=='SVI04']['path'],
            files[files['type']=='SVI05']['path'],
            files[files['type']=='GITCO']['path']
        )

    # unexpected = set(file_types) - viirs_keys
    # if unexpected:
        # logging.warning(f"Warning: Found unexpected file types: {unexpected}")

    if df is None or df.empty:
        return sat, pandas.DataFrame()

    df['key'] = df.merge(files[['path', 'key']], left_on='file_1', right_on='path', how='left')['key']

    # Extract the final list of files
    # results = df.to_numpy().tolist()
    return sat, df

def file_key(file):
    filename = pathlib.Path(file).name
    return "_".join(filename.split('_')[1:6])

def main():
    logging.info("Beginning processing")
    sat, files = load_file_list()
    logging.info(f"Found {len(files)} fileset(s) of type {sat} to process")

    viirs_id = DEVICE_ID_MAP['viirs']
    db = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    for loc in LOCATIONS:
        # Make sure we are using the canonical volcano.
        volc = get_volc(loc)
        elev = volc['elev']
        volc_name = volc['name']

        datastream_mapping = get_datastream_mapping(volc_name)
        if not datastream_mapping:
            logging.warning(f"WARNING: No datastreams found for {volc_name}")
            continue

        start_times = get_start(datastream_mapping)
        start_time = start_times[viirs_id]
        logging.info(f"Found a start time of {start_time} for volcano {volc_name}")
        redis_keys = set(db.keys(f"{volc_name}:*"))
        is_processed = (f"{volc_name}:" + files['key']).isin(redis_keys)
        is_new = files['start_time'] > start_time
        volc_files = files[is_new & ~is_processed]


        if volc_files.empty:
            logging.info(f"No new, unprocessed files to process for {volc_name}")
            logging.info("---------------------")
            continue


        saved_records = 0
        process_idx = 0
        
        # Process in batches, refreshing the ThreadPoolExecutor between each, 
        # to avoid "too many open files" issue.
        BATCH_SIZE = 60
        to_process_all = volc_files.to_numpy().tolist()
        chunks = [to_process_all[i:i + BATCH_SIZE] for i in range(0, len(to_process_all), BATCH_SIZE)]
        for batch_idx, to_process in enumerate(chunks):
            logging.info(f"--- Starting Batch {batch_idx + 1}/{len(chunks)} ---")        
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                future_files = {}
                
                for idx, file_list in enumerate(to_process):
                    fkey = f"{volc_name}:{file_list[-1]}"
                    file_date = file_list[-2]
    
                    logging.debug(f"Submitting {file_list[0].name} with time {file_date} ({idx + 1}/{len(to_process)})")
    
                    future = executor.submit(
                        hotlink_local.get_results,
                        start_time,
                        loc,
                        elev,
                        file_list[:-2],
                        sat
                    )
                    future_files[future] = (file_list[0], fkey)
                    futures.append(future)
    
                logging.info(f"Submitted {len(to_process)} jobs for processing.")
                for future in as_completed(futures):
                    process_idx += 1
                    filename, fkey = future_files.get(future)
                    logging.info(f"Processing results for file {filename} ({process_idx}/{len(to_process_all)})")
                    try:
                        try:
                            results, meta = future.result()
                        except hotlink_local.CoverageError as e:
                            logging.info(f"Insufficient coverage for volcano {volc_name}, {sat}. {e} ({filename})")
                            continue
                        except hotlink_local.AgeError as e:
                            logging.info("File older than most recent results for this location. Skipping.")
                            continue
                        finally:
                            # Mark this file as having been attempted, so we don't try it again
                            exc_type, _, _ = sys.exc_info()
                            if exc_type is None:
                                db.setex(fkey, 129600, "1")
                    except Exception as e:
                        # Log the exception, but don't mark this file as processed.
                        logging.error(f"Unknown exception while processing file: {e}")
                        continue
    
                    if not results.empty and meta['Result Count'] > 0:
                        save_results(results, datastream_mapping)
                        saved_records += 1
                        logging.info(f"Saved results for {volc_name} - {filename}")
                    else:
                        logging.info(f"No results to save for file {filename}, {volc_name} {sat}")
    
                    logging.info(f"Ran HotLINK for {loc}, {filename} ({process_idx}/{len(to_process_all)})")

        logging.info(f"All files processed for volc {volc_name}, saved {saved_records} new records (out of {len(to_process_all)} files) since {start_time}")
        logging.info("----------------------------------")

    logging.info("All files processed.")
    db.close()

if __name__ == "__main__":
    main()
