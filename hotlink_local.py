import functools
import math
import pathlib
import shutil
import time
import warnings

from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, UTC

import hotlink
import numpy
import pandas
import utm

from hotlink import support_functions
from hotlink.preprocess import _gen_output_dir, area_definition
from pyresample import geometry
from satpy import Scene
from tqdm import tqdm

input_path = pathlib.Path('data')

def _gen_output_name(dest, files):
    img_date = datetime.strptime(extract_datetime(files[0]), '%Y%m%dT%H%M%S')
    out_file =  dest / img_date.strftime('%Y%m%d_%H%M.npy')
    return out_file

# Function to get the match key, with substitution for list1
def get_match_key(granule, substitute_prefix=False):
    """
        Get a string to match the associated VIIRS file on. Works with filesystem paths (str or pathlib.Path)
    """

    url = str(granule)
        
    filename = url.split("/")[-1]  # Extract filename from URL
    parts = filename.split("_")
    # Keep everything up to the version, drop processing time and extension
    match_key = "_".join(parts[1:5])  # e.g., j01_d20250528_t1418572_e1420217
    return match_key

def match_viirs(mir_files: list | tuple, tir_files: list | tuple, geog_files: list | tuple) -> pandas.DataFrame:
    """
        Match the VIIRS products, checking the file names since the two result
        lists may not match 1:1
    """
    mir_data = [{
        "file": g,
         "match_key": get_match_key(g, substitute_prefix=True)
    } for g in mir_files]
    
    df = pandas.DataFrame(mir_data)
    if len(df) == 0:
        return df
    
    tir_data = [{
        "file": g,
        "match_key": get_match_key(g, substitute_prefix=False)
    } for g in tir_files]
      
    
    df2 = pandas.DataFrame(tir_data)
    if len(df2) == 0:
        return df2
    
    geog_data = [{
        "file": g,
        "match_key": get_match_key(g, substitute_prefix=False)
    } for g in geog_files]
    
    df_geog = pandas.DataFrame(geog_data)
    if df_geog.empty:
        return df_geog    
    
    df_merged = pandas.merge(df, df2, how="inner", on="match_key", suffixes=("_1", "_2"))
    df_merged = pandas.merge(df_merged, df_geog, how="inner", on="match_key")
    df_merged.rename(columns={"file": "file_3"}, inplace=True)

    return df_merged

def extract_datetime(filename: pathlib.PosixPath) -> str:
    parts = filename.name.split("_")
    date_part = parts[2][1:]  # remove 'd'
    time_part = parts[3][1:7]  # remove 't' and truncate to 6 digits
    
    return f"{date_part}T{time_part}"


def load_and_resample(
    datasets: Sequence[str],
    reader: str,
    area: geometry.AreaDefinition,
    in_files: Sequence,
    out_file: str,
) -> None:
    """
    Load datasets, resample to a specified area, and save the combined data to a file.
    Resampled data file will be in UTM.

    Parameters
    ----------
    datasets : Sequence[str]
        List of dataset names to load and process.
    reader : str
        Reader type used by SatPy to load the datasets.
    area : geometry.AreaDefinition
        The area to which the data should be resampled.
    in_files : Sequence
        List of input file paths containing the datasets.
    out_file : str
        Path to the output file where the resampled data will be saved.

    Returns
    -------
    None
        Output is saved to a Numpy file.

    Notes
    -----
    - source files are deleted after processing.

    Warnings
    --------
    - Warnings related to inefficient chunking operations are suppressed.
    """
    # Loading the scene results in warnings about an ineficient chunking operations
    # Since this is SatPy, and we can't do anything about it, just ignore the warnings.
    warnings.simplefilter("ignore", UserWarning)

    scn=Scene(reader=reader,filenames=[str(f.absolute()) for f in in_files])
    scn.load(datasets,calibration='radiance')

    cropscn = scn.resample(destination=area, datasets=datasets)
  
    mir = cropscn[datasets[0]].to_numpy()
    tir = cropscn[datasets[1]].to_numpy()
    
    total_pixels = mir.size
    valid_pixels = (~numpy.isnan(mir)).sum()
    coverage = (valid_pixels / total_pixels) * 100
    if coverage < .8:
        return

    # Fill missing values
    mir[numpy.isnan(mir)] = numpy.nanmin(mir)
    tir[numpy.isnan(tir)] = numpy.nanmin(tir)

    data = numpy.dstack((mir, tir))
    numpy.save(out_file, data)

    for file in in_files:
        file.unlink()



_process_func: functools.partial = None
def preprocess(
    vent,
    batchsize=200,
    folder='./data',
    output=pathlib.Path('./Output')
):
    global _process_func

    dest = pathlib.Path(folder)
    # lat,lon=vent
    # bounding_box=(float(lon)-0.05,float(lat)-0.05,float(lon)+0.05,float(lat)+0.05)

    meta = {}
    file_types = defaultdict(list)
    files = list(input_path.glob('*.h5'))

    for file in files:
        ftype = file.name.split('_')[0]
        file_types[ftype].append(file)
    
    cols = ['file_1']
    
    if len(file_types.values()) == 3:
        sat = 'viirs'
        df = match_viirs(
            file_types['SVI04'],
            file_types['SVI05'],
            file_types['GITCO']
        )
        cols.append("file_2")
        cols.append("file_3")
    else:
        sat = 'modis'
        df = pandas.DataFrame({'file_1': files,})
        
    if df.empty:
        print("No files found to preprocess after filtering.")
        return {}
    
    df['datetime'] = pandas.to_datetime(
        df['file_1'].apply(extract_datetime)
    )
    
    df = df.sort_values('datetime')
    
    results = df[cols].to_numpy().tolist()

    # We always want batchsize to be even, so VIIRS files will be paired correctly.
    batchsize = batchsize + 1 if batchsize % 2 != 0 else batchsize
    num_results = len(results)
    batches = math.ceil(num_results / batchsize)

    print(f"Found {num_results} files.")

    area=area_definition('name',vent,sat)

    if sat == 'viirs':
        reader = 'viirs_sdr'
        datasets = ['I04','I05']
    else:
        reader = 'modis_l1b'
        datasets = ['21', '32']
    
    _process_func = functools.partial(load_and_resample, datasets, reader, area)
    
    with ProcessPoolExecutor(max_workers=3) as executor:
        # process in batches of no more than batchsize files to save disk space
        # Each file takes around 200MB of space, so 200 files ~=40GB disk space. Processed
        # files are much smaller.
        for k in range(batches):
            # VIIRS files are paired
            if sat=='viirs':
                input_files = results
            else:
                # We run zip here to keep the file list in a consistant format with VIIRS.
                # Each element will be a single-element tuple.
                input_files=zip(dest.glob('M[OY]D0*'))

            input_files = tuple(input_files)

            t1 = time.time()
            print("Beginning resampling of batch.", k + 1)

            futures = [None] * len(input_files) # pre-allocte for a small speedup. Because, why not?
            args = {}

            for idx, files in enumerate(tqdm(
                input_files,
                total = len(input_files),
                desc = "SUBMITTING TASKS",
                unit = "file"
            )):
                out_file =  _gen_output_name(dest, files)
                
                future = executor.submit(_process_func, files, out_file)
                futures[idx] = future
                args[future] = (files, out_file.name)

            # Verify completion of all resampling operations.
            for future in tqdm(
                as_completed(futures),
                total = len(futures),
                desc ="PRE-PROCESSING IMAGES",
                unit = "file"
            ):
                files, out_filename = args[future]

                try:
                    future.result()
                    meta[out_file.name] = {
                        'satelite': files[0].name.split('_')[1],
                        'sensor': sat,
                    }
                except Exception as e:
                    print(f"Unable to process file(s) {files} Exception occured:\n{e}")
                    continue

            print("Resampling of batch", k + 1, "complete in", time.time() - t1, "seconds")

    return meta

def get_results(
    vent: str | tuple[float, float],
    elevation: int,
    out_dir: str | pathlib.Path | None = None
) -> (pandas.DataFrame, dict):

    """
    Retrieve and process satellite images for a given volcano and date range.

    This function downloads satellite images for the specified volcano or vent
    location from the EarthScope database, processes them using the HotLINK
    machine learning model, and returns a pandas DataFrame containing statistical
    results for each processed image.

    Parameters
    ----------
    vent : str | tuple[float, float]
        The name of the volcano (e.g., "Shishaldin") or the coordinates of
        the vent as a tuple (latitude, longitude).
    elevation : int
        The elevation of the vent in meters above sea level.
    dates : tuple[str, str]
        A tuple specifying the start and end dates for data retrieval in the
        format "YYYY-MM-DD" (e.g., `("2023-01-01", "2023-12-31")`).
    sensor : str
        The satellite sensor to retrieve data from. Must be one of:
        - 'viirs': Visible Infrared Imaging Radiometer Suite
        - 'modis': Moderate Resolution Imaging Spectroradiometer
    out_dir : str | Path, default "Output/{sensor}"
        The directory in which to save output image products. Will be created
        if it does not exist.

    Returns
    -------
    results: pandas.DataFrame
        A DataFrame containing the processed results for each image. Each row
        corresponds to an  input image and includes model output.
    meta: dict
        A Dictionary containing metadata about the run

    Raises
    ------
    ValueError
        If the specified volcano name is not found in the volcano database.

    Notes
    -----
    - The function determines the volcano location based on the name or coordinates.
      If a name is provided, it searches for the volcano in the internal database.
      If coordinates are provided, it finds the nearest volcano to the given location.

    Examples
    --------
    >>> results = get_results(
    ...     vent="Shishaldin",
    ...     elevation=2550,
    ...     dates=("2019-01-01", "2019-12-31"),
    ...     sensor="viirs",
    ...     out_dir="Output Images"
    ... )
    >>> print(results)

    >>> results = get_results(
    ...     vent=(54.7554, -163.9711),
    ...     elevation=2550,
    ...     dates=("2019-01-01", "2019-12-31"),
    ...     sensor="viirs"
    ... )
    >>> print(results)
    """
    sensor = 'VIIRS'    

    meta = {
        'Vent': vent,
        'Elevation': elevation,
#        'Data Dates': dates,
#        'Sensor': sensor,
        'Run Start': datetime.now(UTC).isoformat(),
    }

    volcs = support_functions.load_volcanoes()

    if isinstance(vent, str):
        volc = volcs[volcs['name']==vent]
        if len(volc) == 0:
            raise ValueError("Specified volcano not found!")
        vent = (volc.iloc[0]['lat'], volc.iloc[0]['lon'])
    else:
        dists = support_functions.haversine_np(vent[1], vent[0], volcs['lon'], volcs['lat'])
        volcs.loc[:, 'dist'] = dists
        volc = volcs[volcs['dist']==volcs['dist'].min()]

    print("Using volcano:", volc.iloc[0]['name'], "location:", vent)

    meta['Volcano Name'] = volc.iloc[0]['name']
    meta['Volcano ID'] = volc.iloc[0]['id']
    meta['Center'] = vent

    if out_dir is None:
        out_dir = pathlib.Path("Output") / sensor

    # Make sure this is a pathlib.Path object, and make sure it exists, creating it if needed.
    output_dir = pathlib.Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents = True, exist_ok = True)

    data_path = pathlib.Path('./data')

    # make sure the data directory exists
    data_path.mkdir(exist_ok = True)

    print("Searching for files to download...")
    download_meta = preprocess(
        vent,
        folder = data_path,
        output=output_dir
    )

    print("Image files processed. Beginning calculations")

    # Set some constants based on sensor
    if sensor.upper() == 'MODIS':
        RES = 1000
        MIR_WL = support_functions.MODIS_MIR_WL
        TIR_WL = support_functions.MODIS_TIR_WL
        RP_CONSTANT = support_functions.MODIS_MIR_RP_CONSTANT
    else:
        # VIIRS
        RES = 375
        MIR_WL = support_functions.VIIRS_MIR_WL
        TIR_WL = support_functions.VIIRS_TIR_WL
        RP_CONSTANT = support_functions.VIIRS_MIR_RP_CONSTANT

    # Calculate the GeoTransform for output images
    center_x, center_y, utm_zone, utm_lat_band = utm.from_latlon(*vent)
    # resolution = 1000 if sensor.upper() == 'MODIS' else 375
    # size = 24
    # transform = rasterio.transform.from_origin(center_x - (size / 2) * resolution,
                                               # center_y + (size / 2) * resolution,
                                               # resolution, resolution)

    # hemisphere = hemisphere = " +south" if utm_lat_band < 'N' else ""

    # crs = f"+proj=utm +zone={utm_zone}{hemisphere} +datum=WGS84 +units=m +no_defs"
    meta['UTM Zone'] = utm_zone
    meta['UTM Latitude Band'] = utm_lat_band
    
    # TODO: split into VIIRS and MODIS. Process separately (or come up with better way)

    data_files = list(data_path.glob('*.npy'))

    if not data_files:
        print("WARNING: No data files to process.")
        # Define the expected columns for an empty DataFrame
        expected_columns = [
            'Data File', 'Number Hotspot Pixels', 'Hotspot Radiative Power (W)',
            'MIR Hotspot Brightness Temperature', 'MIR Background Brightness Temperature',
            'MIR Hotspot Max Brightness Temperature', 'TIR Hotspot Brightness Temperature',
            'TIR Background Brightness Temperature', 'TIR Hotspot Max Brightness Temperature',
            'Day/Night Flag', 'Solar Zenith', 'Solar Azimuth', 'Date', 'Max Probability',
            'Pixels Above 0.5 Probability', 'Sensor', 'Volcano ID', 'Satellite', 'Data URL'
        ]
        # Return an empty DataFrame with the expected structure
        empty_results = pandas.DataFrame(columns=expected_columns)
        # Update meta with the failure reason and end time
        meta['Result Count'] = 0
        meta['Error'] = "No .npy files found in the data directory"
        meta['Run End'] = datetime.now(UTC).isoformat()
        return empty_results, meta

    model = hotlink.process.load_hotlink_model()

    img_data, img_dates = hotlink.process.load_data_files(data_files)
    # Make sure there are no missing pixels
    mir_data = img_data[:, :, :, 0] # creates a view
    tir_data = img_data[:, :, :, 1]

    # Create masks for NaN values
    nan_mask_tir = numpy.isnan(tir_data)
    nan_mask_mir = numpy.isnan(mir_data)

    # Fill NaN values in the mir/tir arrays with min values for the array
    if nan_mask_mir.any():
        # create arrays with the minimum value for each image
        min_mir_observed = numpy.nanmin(mir_data, axis=(1, 2), keepdims=True)
        min_mir_observed = numpy.broadcast_to(min_mir_observed, mir_data.shape)
        # Fill NaN values with the corresponding minimum values
        mir_data[nan_mask_mir] = min_mir_observed[nan_mask_mir]

    if nan_mask_tir.any():
        min_tir_observed = numpy.nanmin(tir_data, axis=(1, 2), keepdims=True)
        min_tir_observed = numpy.broadcast_to(min_tir_observed, tir_data.shape)
        tir_data[nan_mask_tir] = min_tir_observed[nan_mask_tir]

    mir_analysis = support_functions.crop_center(mir_data, size=24, crop_dimensions=(1, 2))
    tir_analysis = support_functions.crop_center(tir_data, size=24, crop_dimensions=(1, 2))

    mir_bt = support_functions.brightness_temperature(mir_analysis, wl=MIR_WL)
    tir_bt = support_functions.brightness_temperature(tir_analysis, wl=TIR_WL)

    n_data = img_data.copy()
    n_data[:, :, :, 0] = support_functions.normalize_MIR(n_data[:, :, :, 0])
    n_data[:, :, :, 1] = support_functions.normalize_TIR(n_data[:, :, :, 1])

    predict_data = support_functions.crop_center(n_data, crop_dimensions=(1, 2))
    predict_data = predict_data.reshape(n_data.shape[0], 64, 64, 2)

    print("Predicting hotspots...")
    prediction = model.predict(predict_data) #shape=[batch_size, 24, 24, 3], for 3 predicted classes:background, hotspot-adjacent, and hotspot

    # use hysteresis thresholding to generate a binary map of hotspot pixels
    prob_active = prediction[:,:,:,2] #map with probabilities of active class

    #highest probability per image, equated to probability that the image contains a hotspot
    max_prob = numpy.round(numpy.max(prob_active, axis=(1, 2)), 3)
    prob_above_05 = numpy.count_nonzero(prob_active>0.5, axis=(1, 2))

    process_progress = tqdm(
        total=img_data.shape[0],
        desc="CALCULATING RESULTS"
    )

    def _run_calcs(idx):
        result = {}
        img_file = data_files[idx]
        image_date = img_dates[idx]

        result['Data File'] = img_file.name

        hotspot_mask = hotlink.process.apply_hysteresis_threshold(prob_active[idx], low=0.4, high=0.5).astype('bool')
        hotspot_pixels = numpy.count_nonzero(hotspot_mask)
        result['Number Hotspot Pixels'] = hotspot_pixels

        rp = support_functions.radiative_power(
            mir_analysis[idx],
            hotspot_mask,
            cellsize=RES,
            rp_constant=RP_CONSTANT
        ) if hotspot_mask.any() else 0

        result['Hotspot Radiative Power (W)'] = round(rp, 4)

        # mir hotspot/background brightness temperature analysis
        hotspot_mir_bt = mir_bt[idx][hotspot_mask]
        bg_mir_bt = mir_bt[idx][~hotspot_mask]

        result['MIR Hotspot Brightness Temperature'] = hotspot_mir_bt.mean().round(4)
        result['MIR Background Brightness Temperature'] = bg_mir_bt.mean().round(4)
        result['MIR Hotspot Max Brightness Temperature'] = hotspot_mir_bt.max().round(4) if hotspot_mir_bt.size > 0 else numpy.nan

        # tir hotspot/background brigbhtness temerature analysis
        hotspot_tir_bt = tir_bt[idx][hotspot_mask]
        bg_tir_bt = tir_bt[idx][~hotspot_mask]

        result['TIR Hotspot Brightness Temperature'] = hotspot_tir_bt.mean().round(4)
        result['TIR Background Brightness Temperature'] = bg_tir_bt.mean().round(4)
        result['TIR Hotspot Max Brightness Temperature'] = hotspot_tir_bt.max().round(4) if hotspot_tir_bt.size > 0 else numpy.nan

        day_night = support_functions.get_dn(image_date, vent[1], vent[0], elevation)
        sol_zenith, sol_azimuth = support_functions.get_solar_coords(
            image_date, vent[1], vent[0], elevation
        )

        result['Day/Night Flag'] = day_night
        result['Solar Zenith'] = round(sol_zenith, 1)
        result['Solar Azimuth'] = round(sol_azimuth, 1)

        process_progress.update()
        return result

    # Not sure if this is really needed, as this loop is fast, but might
    # speed things up a bit.
    with ThreadPoolExecutor() as executor:
        results = executor.map(_run_calcs, range(img_data.shape[0]))

    results = pandas.DataFrame(results)
    results.reset_index(drop=True, inplace=True)

    results['Date'] = img_dates
    results['Max Probability'] = max_prob
    results['Pixels Above 0.5 Probability'] = prob_above_05

    # Single values apply to all records
    results['Sensor'] = sensor.upper()
    results['Volcano ID'] = volc.iloc[0]['id']

    # pull in metadata retrieved during the download
    file_meta = results['Data File'].map(lambda x: download_meta.get(x, {}))
    results['Satellite'] = file_meta.map(lambda x: x.get('satelite'))

    SAVE_IMAGES = False # TODO: make this a user passable flag somewhere.

    for idx, (image_date, img_file) in tqdm(
        enumerate(zip(img_dates, data_files)),
        total=len(img_dates),
        unit="IMAGES",
        desc="SAVING IMAGES"
    ):
        if SAVE_IMAGES:
            ########## IMAGE SAVE/Data File Archive #################
            # This section deals with saving PNG images and archiving
            # the pre-processed data files. Remove this section if not
            # desired
            #########################################################

            # Save the .png images. Second loop, but this one doesn't lend itself to
            # parallel processing at all.
            file_out_dir = _gen_output_dir(img_file, out_dir)
            file_out_dir.mkdir(parents=True, exist_ok=True)

            # Save MIR images
            mir_image = file_out_dir / f"{img_file.stem}_mir.png"
            results.loc[idx, 'MIR Image'] = str(mir_image)

            hotlink.process._save_fig(
                mir_data[idx],
                mir_image,
                f"Middle Infrared\n{image_date.strftime('%Y-%m-%d %H:%M')}"
            )

            # slice_prob_active = prob_active[idx]

            # Optional: save probability GeoTIFF (currently disabled)
            # NOTE: These files are EXTREAMLY tiny at only 24px x 24px
            # geotiff_file = output_dir / f"{img_file.stem}_probability.tif"
            # result['Probability TIFF'] = str(geotiff_file)

            # with rasterio.open(
                # geotiff_file,
                # 'w',
                # driver = 'GTiff',
                # height = slice_prob_active.shape[0],
                # width = slice_prob_active.shape[1],
                # count = 1,
                # dtype = slice_prob_active.dtype,
                # crs=crs,
                # transform=transform
            # ) as dst:
                # dst.write(slice_prob_active, 1)

            # Move the processed data file to the output directory
            shutil.move(str(img_file), str(file_out_dir / img_file.name))
            ###################### END IMAGE SECTION ###########################

        img_file.unlink(missing_ok=True)

    meta['Result Count'] = len(results)

    if len(results) > 0:
        results = results.sort_values('Date').reset_index(drop = True)

    meta['Run End'] = datetime.now(UTC).isoformat()
    return results, meta
