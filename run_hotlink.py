import json
import time

import psycopg

from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache

import config
import hotlink
import pandas

from hotlink import support_functions

########## CONSTANTS #########
LOCATIONS = ['Spurr']

# dict to map the output column name to database variable name
VARIABLE_NAME_MAP = {
    "Max Probability": "HotLINK probability",
    "Number Hotspot Pixels": "Number of Hotspot Pixels",
    "Hotspot Radiative Power (W)": "Radiative Power, W",
    "MIR Background Brightness Temperature": "Background MIR Brightness Temperature",
    "MIR Hotspot Brightness Temperature": "Hotspot MIR Mean Brightness Temperature",
    "MIR Hotspot Max Brightness Temperature": "Hotspot MIR Maximum Brightness Temperature",
    "TIR Hotspot Brightness Temperature": "Hotspot TIR mean Brightness Temperature",
    "TIR Hotspot Max Brightness Temperature": "Hotspot TIR Maximum Brightness Temperature",
    "Solar Zenith": "Solar Zenith Angle",
    "Solar Azimuth": "Solar Azimuth Angle",
    # Metadata needs special handling (see below)
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
            variables.variable_name,
            devices.device_name
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
    IV_MAP = {v: k for k, v in VARIABLE_NAME_MAP.items()}    
    
    mapping = {
        (IV_MAP.get(row[1], row[1]),
         row[2].lower()
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
        
   
def get_start(datastreams):
    # Group datastream_ids by sensor
    sensor_ids = {"viirs": [], "modis": []}
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
                
            
def run_hotlink(loc, elev, dates, sensor, mapping):
    results, meta = hotlink.get_results(loc, elev, dates, sensor)
    
    # Save results to PREEVENTS database
    with preevents_cursor(readonly=False) as cursor:
        for _, row in results.iterrows():
            sensor = row["Sensor"].lower() # Pull from results for good measure
            timestamp = row["Date"]
            
            # Save the metadata record
            metadata = {"day_night": row["Day/Night Flag"], "satellite": row["Satellite"], "sensor": row["Sensor"]}
            metadata_json = json.dumps(metadata)
            metadata_datastream = mapping.get(("Metadata", sensor))
            if metadata_datastream:
                cursor.execute(
                    """INSERT INTO datavalues (
                        datastream_id, timestamp, categoryvalue, last_updated
                    )
                    VALUES (%s, %s, %s, now())
                    """,
                    (metadata_datastream, timestamp, metadata_json)
                )
            # Save the value records
            for result_key in VARIABLE_NAME_MAP.keys():
                if result_key in row and not pandas.isna(row[result_key]):
                    key = (result_key, sensor)
                    datastream_id = mapping.get(key)
                    if datastream_id:
                        cursor.execute(
                            """
                            INSERT INTO datavalues (datastream_id, timestamp, datavalue, last_updated)
                            VALUES (%s, %s, %s, now())
                            """,
                            (datastream_id, timestamp, float(row[result_key]))
                        )
                    else:
                        print(f"Warning: No datastream_id for {result_key} with sensor {sensor}")
        cursor.connection.commit()
    

def main():
    end_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:00')
    for loc in LOCATIONS:
        t1 = time.time()        
        
        # Make sure we are using the canonical volcano.
        volc = get_volc(loc)
        elev = volc['elev']
        volc_name = volc['name']
        
        datastream_mapping = get_datastream_mapping(volc_name)
        start_times = get_start(datastream_mapping)
        
        for sensor in ['viirs', 'modis']:
            start_time = start_times[sensor].strftime('%Y-%m-%dT%H:%M:00')
            dates = (start_time, end_time)
            print(f"****Running {volc_name} {sensor} for {dates}")
            
            run_hotlink(volc_name, elev, dates, sensor, datastream_mapping)
        print(f"Ran HotLINK for {loc} in {time.time() - t1} seconds")
            
if __name__ == "__main__":
    main()