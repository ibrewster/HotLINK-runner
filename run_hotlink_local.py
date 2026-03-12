import time

import matplotlib
import numpy
import requests

matplotlib.use('Agg')

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
from functools import lru_cache
from io import BytesIO
from os.path import splitext

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import pandas
import psycopg
import utm

from pyproj import Transformer, CRS
from pyresample import create_area_def
from pyresample.geometry import AreaDefinition
from satpy import Scene

from hotlink import support_functions
from hotlink.preprocess import area_definition

import matplotlib.pyplot as plt

import config
import hotlink_local
import mattermost
from utils import preevents_cursor, REDIS_DB

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
    "Metadata": 13,    # Metadata needs special handling (see below)
    "MIR Image": 1,
}

DEVICE_ID_MAP = {
    'viirs': 1,
    'modis': 2,
}

#############################



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

def generate_mir_image(img, title, volc):
    # Convert img from Kelven to ºC
    img -= 273.15

    volc_lon, volc_lat =  volc[['lon', 'lat']]
    volc_area: AreaDefinition = area_definition(volc['id'], (volc_lat, volc_lon), 'viirs')

    _, _, zone_number, zone_letter = utm.from_latlon(volc_lat, volc_lon)
    utm_crs = ccrs.UTM(zone=zone_number, southern_hemisphere=(volc_lat < 0))

    x0, y0, x1, y1 = volc_area.area_extent  # (x_ll, y_ll, x_ur, y_ur)

    fig, ax = plt.subplots(
        figsize=(5.3, 4),
        subplot_kw={"projection": utm_crs} # display in UTM
    )

    ax.set_extent([x0, x1, y0, y1], crs=utm_crs)

    im = ax.imshow(
        img,
        origin="upper",
        extent=[x0, x1, y0, y1],
        transform=utm_crs,
        interpolation="nearest",
        vmin=-50,
        vmax=50
    )

    fig.colorbar(im, ax=ax, shrink=0.7, label="Brightness Temperature (ºC)")

    # Transformer to convert UTM ticks to lat/lon for labels
    transformer = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    volc_x, volc_y = transformer.transform(
        volc_lon,
        volc_lat,
        direction = 'INVERSE'
    )

    x_ticks = numpy.linspace(x0, x1, 3)
    y_ticks = numpy.linspace(y0, y1, 3)

    x_ticks[1] = volc_x
    y_ticks[1] = volc_y

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # For x labels: convert each (x, mid_y) to lon
    mid_y = (y0 + y1) / 2
    x_lons = [transformer.transform(x, mid_y)[0] for x in x_ticks]
    x_lons[1] = volc_lon
    ax.set_xticklabels([f"{lon:.3f}°E" for lon in x_lons])

    # For y labels: convert each (mid_x, y) to lat
    mid_x = (x0 + x1) / 2
    y_lats = [transformer.transform(mid_x, y)[1] for y in y_ticks]
    y_lats[1] = volc_lat
    ax.set_yticklabels([f"{lat:.3f}°N" for lat in y_lats])

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.set_title(title)
    fig.tight_layout()

    out = BytesIO()
    fig.savefig(out, format="png", bbox_inches='tight')

    out.seek(0)
    plt.close()

    return out

def post_mattermost(img, volcano_id, filename, meta):
    VOLCS = load_volcs()
    volcano = VOLCS.loc[VOLCS['id'] == volcano_id, 'name'].iloc[0]

    message = f"""### {volcano} HotLINK Detection.
**Image Date:** {meta['Date'].strftime('%m/%d/%Y %H:%M')} UTC

**Max Probability:** {round(meta['Max Probability'] * 100)}%
**Satelite:** {meta['Satellite']}"""

    matt, channel = mattermost.connect()
    msg_meta = mattermost.mm_upload(matt, channel, message, image=img, img_name=filename)
    return msg_meta

def save_results(results, mapping):
    # Save results to PREEVENTS database
    VOLCS = load_volcs() # Cached, so essentially just a dictionary lookup
    results['Day/Night Flag'] = results['Day/Night Flag'].apply(lambda x: json.dumps({"day_night": x}))
    with preevents_cursor(readonly=False) as cursor:
        for _, row in results.iterrows():
            sensor = row["Sensor"].lower() # Pull from results for good measure
            sensor = DEVICE_ID_MAP[sensor]
            timestamp = row["Date"]

            # Save the MIR Image
            mir_data = row['MIRImage']
            img_volc = VOLCS.loc[VOLCS['id']==row["Volcano ID"]].iloc[0]
            mir_filename = f"{img_volc['name']}-{splitext(row['Data File'])[0]}_mir.png"
            mir_title = f"{img_volc['name']} Middle Infrared\n{timestamp.strftime('%Y-%m-%d %H:%M')} UTC"
            img_bytes = generate_mir_image(mir_data, mir_title, img_volc)

            metadata = {"satellite": row["Satellite"], "sensor": row["Sensor"]}

            if row['Max Probability'] >= 0.5 and json.loads(row['Day/Night Flag'])['day_night'] == 'N':
                msg_meta = post_mattermost(img_bytes, row['Volcano ID'], mir_filename, row)
                msg_id = msg_meta['id']
                metadata['mattermostid'] = msg_id

            # Upload the image to the PREEVENTS server
            img_bytes.seek(0)
            upload_resp = requests.post(
                'https://preeventsdb.gi.alaska.edu/api/v1/uploads',
                files={'file': (mir_filename, img_bytes, 'image/png')},
                headers={'X-API-Key': config.PREEVENTS_UPLOAD_KEY}
            )
            if upload_resp.status_code == 200:
                up_resp = upload_resp.json()
                row['MIR Image'] = up_resp['upload_id']

            # Save the metadata record
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

def load_file_list() -> pandas.DataFrame:
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

    if df is None or df.empty:
        return sat, pandas.DataFrame()

    return sat, df

def file_key(file):
    filename = pathlib.Path(file).name
    return "_".join(filename.split('_')[1:6])

def process_volc(loc: str|list, orbit, db, scn_albers, sat):
    volc_info = get_volc(loc)
    elev = volc_info['elev']
    volc_name = volc_info['name']

    datastream_mapping = get_datastream_mapping(volc_name)
    if not datastream_mapping:
        logging.warning(f"WARNING: No datastreams found for {volc_name}")
        return None, volc_name, None  # caller checks for None

    redis_keys = set(db.keys(f"{volc_name}:*"))
    if f"{volc_name}:{orbit}" in redis_keys:
        logging.info(f"Orbit {orbit} has already been processed for volcano {volc_name}. Skipping.")
        return None, volc_name, None

    result = hotlink_local.get_results(loc, elev, scn_albers, sat)
    return result, volc_name, datastream_mapping


def debug_dump_swath(scn, output_path="debug_i04_swath.png"):
    i04 = scn["I04"]
    data = i04.values

    # Lon/lat live on the swath geometry, not as xarray coords
    swath_def = i04.attrs["area"]
    import dask.array

    lons, lats = swath_def.get_lonlats()
    lons = dask.array.Array.compute(lons)
    lats = dask.array.Array.compute(lats)

    fig, ax = plt.subplots(
        figsize=(12, 10),
        subplot_kw={"projection": ccrs.AlbersEqualArea(
            central_longitude=-154,
            central_latitude=50,
            standard_parallels=(55, 65),
        )}
    )

    vmin, vmax = numpy.nanpercentile(data, [2, 98])

    ax.pcolormesh(
        lons, lats, data,
        transform=ccrs.PlateCarree(),
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="gray")
    ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", color="gray")

    ax.set_global()

    lat_min, lat_max = numpy.nanmin(lats), numpy.nanmax(lats)
    lon_min, lon_max = numpy.nanmin(lons), numpy.nanmax(lons)

    print(f"lat range: {lat_min:.2f} to {lat_max:.2f}")
    print(f"lon range: {lon_min:.2f} to {lon_max:.2f}")

    ax.set_title(
        f"I04 swath extent (pre-resample)\n"
        f"lat [{lat_min:.2f} → {lat_max:.2f}]  lon [{lon_min:.2f} → {lon_max:.2f}]"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")
    
    
def debug_dump_i04(scn_albers, area_def, output_path="debug_i04.png"):
    """
    Dump the I04 (MIR) band from an Albers-resampled satpy scene to an image.
    """
    i04 = scn_albers["I04"]
    data = i04.values  # numpy array, may contain NaNs
    
    # Grab the projection from the area_def
    proj_dict = area_def.proj_dict
    crs = ccrs.AlbersEqualArea(
        central_longitude=proj_dict.get("lon_0", 0),
        central_latitude=proj_dict.get("lat_0", 0),
        standard_parallels=(
            proj_dict.get("lat_1", 29.5),
            proj_dict.get("lat_2", 45.5),
        )
    )
    
    # Area extent in projection coordinates
    extent = [
        area_def.area_extent[0],  # x_min
        area_def.area_extent[2],  # x_max
        area_def.area_extent[1],  # y_min
        area_def.area_extent[3],  # y_max
    ]
    
    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw={"projection": crs}
    )
    
    # Plot with a fire-appropriate colormap; clip to reasonable radiance range
    vmin, vmax = numpy.nanpercentile(data, [2, 98])
    im = ax.imshow(
        data,
        origin="upper",
        extent=extent,
        transform=crs,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )
    
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="gray")
    
    plt.colorbar(im, ax=ax, label="Radiance (W·m⁻²·sr⁻¹·μm⁻¹)", shrink=0.7)
    ax.set_title(f"I04 MIR Band — orbit debug dump\n"
                 f"shape: {data.shape}  |  valid px: {numpy.sum(~numpy.isnan(data)):,}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved debug image → {output_path}")

def main():
    t0 = time.time()
    logging.info("Beginning processing")
    sat, files = load_file_list()
    logging.info(f"Found {len(files)} fileset(s) of type {sat} to process")
    
    # Create an area definition covering "Alaska"
    crs = CRS.from_proj4(
        "+proj=aea +lat_0=50 +lon_0=-154 +lat_1=55 +lat_2=65 +datum=NAD83 +units=m"
    )
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    corners = [
        (172.0, 50.0),   # SW — far western Aleutians
        (172.0, 60.0),   # NW
        (-129.0, 50.0),  # SE — Alaska panhandle
        (-129.0, 60.0),  # NE
    ]
    xs, ys = transformer.transform(
        [c[0] for c in corners],
        [c[1] for c in corners],
    )

    area_def = create_area_def(
        "alaska_albers_operational",
        crs,
        area_extent=[min(xs), min(ys), max(xs), max(ys)],
        resolution=371,
        units="m",
    )
    # area_def = AreaDefinition.from_epsg(3338, resolution=371)

    with ThreadPoolExecutor(max_workers=4) as executor:
        orbit_groups = files.groupby("orbit", sort=False)
        for orbit, group in orbit_groups:
            redis_key = f"processed:{orbit}"
            if REDIS_DB.exists(redis_key):
                logging.info(f"Orbit {orbit} already processed. Skipping")
                continue

            if group.empty:
                logging.info(f"No files to process for orbit {orbit}")
                continue

            all_processed = True
            logging.info(f"Loading files for orbit {orbit}")
            file_list = list(
                group[["file_1", "file_2", "file_3"]]
                .stack()
            )

            scn=Scene(reader='viirs_sdr',filenames=[str(f.absolute()) for f in file_list])
            datasets = ['I04','I05']
            scn.load(datasets,calibration='radiance')

            logging.info(f"Resampling and loading scene for orbit {orbit}")
            # By resampling to albers here, and then loading the resampled datasets (next)
            # we can extract the AOI for each volcano in seconds, rather than
            # having to spend minutes per volcano re-loading the data for the entire swath
            scn_albers = scn.resample(area_def, datasets=['I04', 'I05'])

            for dataset in datasets:
                t1 = time.time()
                scn_albers[dataset].load()
                logging.info(f"Loaded dataset {dataset} in {time.time() - t1}")

            futures = []
            future_files = {}

            for loc in LOCATIONS:
                logging.debug(f"Submitting {orbit} for {loc}")
                future = executor.submit(
                    process_volc,
                    loc,
                    orbit,
                    REDIS_DB,
                    scn_albers,
                    sat
                )

                future_files[future] = (orbit, loc)
                futures.append(future)

            logging.info(f"Submitted {len(futures)} jobs for processing.")
            saved_records = 0
            process_idx = 0
            for future in as_completed(futures):
                process_idx += 1
                orbit, volc = future_files.get(future)
                logging.info(f"Processing results for {volc}, orbit {orbit} ({process_idx}/{len(futures)})")

                mark_processed = True
                try:
                    hotlink_result, volc, datastreams = future.result()
                    if hotlink_result is None:
                        continue

                    results, meta = hotlink_result
                except hotlink_local.CoverageError as e:
                    logging.info(f"Insufficient coverage for volcano {volc}, {sat}. {e} (orbit {orbit})")
                    continue
                except hotlink_local.AgeError as e:
                    logging.info(f"Orbit {orbit} older than most recent results for {volc}. Skipping.")
                    continue
                except Exception as e:
                    # Log the exception, but don't mark this file as processed.
                    logging.exception(f"Unknown exception while processing {volc}, orbit {orbit}: {e}")
                    mark_processed = False
                    all_processed = False
                    continue
                finally:
                    # Mark this file as having been attempted, so we don't try it again
                    exc_type, _, _ = sys.exc_info()
                    if exc_type is None and mark_processed:
                        REDIS_DB.setex(f"{volc}:{orbit}", 129600, "1")

                if not results.empty and meta['Result Count'] > 0:
                    # Add the orbit number to the results
                    results['orbit'] = orbit
                    save_results(results, datastreams)

                    saved_records += 1
                    logging.info(f"Saved results for {volc} - orbit {orbit}")
                else:
                    logging.info(f"No results to save for orbit {orbit}, {volc} {sat}")

                logging.info(f"Ran HotLINK for {volc}, orbit {orbit} ({process_idx}/{len(futures)})")
                logging.info("----------------------------------")

            if all_processed:
                REDIS_DB.setex(redis_key, 129600, "1")
            logging.info(f"All locations processed for orbit {orbit}, saved {saved_records} new records (out of {len(futures)} locations)")

    logging.info(f"All orbits processed in {time.time()-t0}.")
    REDIS_DB.close()

if __name__ == "__main__":
    main()
