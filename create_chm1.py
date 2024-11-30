import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from rasterio.mask import mask
import rasterio
from rasterio.transform import from_origin
import laspy

def get_raster_bounding_box(raster_file):
    """Get the bounding box of the raster file"""
    with rasterio.open(raster_file) as src:
        # Get the bounding box coordinates
        bounds = src.bounds
        return bounds

# def chip_raster_bounding_box(raster_file):
#     bbox= get_raster_bounding_box(raster_file)
#     """Divide the bounding box into four equal quadrants"""
#     ul_lon, lr_lon = bbox.left, bbox.right
#     ul_lat, lr_lat = bbox.top, bbox.bottom

#     # Calculate midpoints
#     mid_lon = (ul_lon + lr_lon) / 2
#     mid_lat = (ul_lat + lr_lat) / 2

#     # Define the bounding boxes for each quadrant
#     quadrant_bboxes = [(ul_lon, mid_lon, ul_lat, mid_lat),(mid_lon, lr_lon, ul_lat, mid_lat),(ul_lon, mid_lon, mid_lat, lr_lat),(mid_lon, lr_lon, mid_lat, lr_lat)]

#     return quadrant_bboxes


def chip_raster_bounding_box(raster_file, rows, cols):
    """Divide the bounding box into a grid of equal-sized chips"""
    bbox = get_raster_bounding_box(raster_file)
    ul_lon, lr_lon = bbox.left, bbox.right
    ul_lat, lr_lat = bbox.top, bbox.bottom

    # Calculate step sizes for longitude and latitude
    lon_step = (lr_lon - ul_lon) / cols
    lat_step = (ul_lat - lr_lat) / rows

    # Define the bounding boxes for each chip
    chips = []
    for i in range(rows):
        for j in range(cols):
            # Calculate the coordinates for the chip
            chip_ul_lon = ul_lon + j * lon_step
            chip_lr_lon = ul_lon + (j + 1) * lon_step
            chip_ul_lat = ul_lat - (i + 1) * lat_step
            chip_lr_lat = ul_lat - i * lat_step
            chips.append((chip_ul_lon, chip_lr_lon, chip_ul_lat, chip_lr_lat))
    return chips

def ply2las(points, to_extract, output_path, offset):
    data_pts = []
    to_extract = [to_extract] if not isinstance(to_extract, list) else to_extract
    for i in to_extract:
        idx = np.where(points[:, -1] == i)
        data_pts.append(points[idx])
    tree_points = np.vstack(data_pts)
    try:
        xmin = np.min(tree_points[:, 0])
    except:
        return 0

    if offset == 0:
        xoff = np.floor(np.min(tree_points[:, 0]))
        yoff = np.floor(np.min(tree_points[:, 1]))
        zoff = np.floor(np.min(tree_points[:, 2]))
    else:
        xoff = offset[0]
        yoff = offset[1]
        zoff = offset[2]

    x = tree_points[:, 0] - xoff
    y = tree_points[:, 1] - yoff
    z = tree_points[:, 2] - zoff

    xmin = np.min(x)
    ymin = np.min(y)
    zmin = np.min(z)
    xmax = np.max(x)
    ymax = np.max(y)
    zmax = np.max(z)

    intensity = tree_points[:, 3]
    classif = tree_points[:, 4]

    output_path = Path(output_path)

    header = laspy.header.LasHeader()
    outfile = laspy.LasData(header)
    outfile.header.minimum = [xmin, ymin, zmin]
    outfile.header.maximum = [xmax, ymax, zmax]
    outfile.header.offset = [xoff, yoff, zoff]
    outfile.x = x
    outfile.y = y
    outfile.z = z
    outfile.intensity = intensity
    outfile.classif = classif
    outfile.write(output_path)
    return outfile

def get_bounding_boxes(las_data, num_chips=2):
    """Divide the bounding box of the LAS data into smaller chips.
    
    Parameters
    ----------
    las_data : GeoDataFrame
        The LAS data as a GeoDataFrame.
    num_chips : int
        Number of chips to divide the data into (2 for each dimension by default).
    
    Returns
    -------
    list of tuples
        List of bounding boxes (lon_min, lon_max, lat_min, lat_max).
    """
    lon_min, lon_max, lat_min, lat_max = las_data.total_bounds
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    # Calculate step size
    lon_step = lon_range / num_chips
    lat_step = lat_range / num_chips

    bounding_boxes = []
    for i in range(num_chips):
        for j in range(num_chips):
            bbox = (
                lon_min + i * lon_step,
                lon_min + (i + 1) * lon_step,
                lat_min + j * lat_step,
                lat_min + (j + 1) * lat_step
            )
            bounding_boxes.append(bbox)

    return bounding_boxes

def resize_dsm_to_dem(dsm_path, dem_path, resized_dsm_path):
    with rasterio.open(dsm_path) as dsm, rasterio.open(dem_path) as dem:
        # Define target resolution and transform from DEM
        target_resolution = dem.res
        target_transform = dem.transform
        target_crs = dem.crs

        # Define the new width and height based on DEM's bounds and resolution
        new_width = int((dem.bounds.right - dem.bounds.left) / target_resolution[0])
        new_height = int((dem.bounds.top - dem.bounds.bottom) / target_resolution[1])

        # Update DSM profile with new dimensions and transform
        dsm_profile = dsm.profile.copy()
        dsm_profile.update({
            'height': new_height,
            'width': new_width,
            'transform': target_transform,
            'crs': target_crs,
            'dtype': 'float32'  # Ensure the data type is suitable for floating-point operations
        })

        # Resample DSM to match DEM's resolution and bounds
        with rasterio.open(resized_dsm_path, 'w', **dsm_profile) as resized_dsm:
            for i in range(1, dsm.count + 1):
                data = dsm.read(i)
                # Create an empty array to hold the resampled data
                resampled_data = np.empty((new_height, new_width), dtype=data.dtype)
                
                # Resample DSM data to match DEM resolution
                reproject(
                    source=data,
                    destination=resampled_data,
                    src_transform=dsm.transform,
                    src_crs=target_crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                ) # Get NoData value from DSM
                nodata = dsm.nodata

                if nodata is not None:
                    # Read the DEM data
                    dem_data = dem.read(1)

                    # Identify where DSM data is NoData
                    nodata_mask = np.isnan(resampled_data) if nodata is np.nan else (resampled_data == nodata)

                    # Replace NoData values in resampled DSM with values from DEM
                    resampled_data[nodata_mask] = dem_data[nodata_mask]
                resized_dsm.write(resampled_data, i)

    print(f"DSM successfully resized to match DEM resolution and saved to {resized_dsm_path}")

def create_chm(dsm_path, dem_path, resized_dsm_path, chm_path):
    # Resize DSM to match DEM
    resize_dsm_to_dem(dsm_path, dem_path, resized_dsm_path)

    with rasterio.open(resized_dsm_path) as resized_dsm, rasterio.open(dem_path) as dem:
        # Read data
        dsm_data = resized_dsm.read(1)
        dem_data = dem.read(1)
        
        # Mask for null values
        mask_null = (dsm_data == -9999) | (dem_data == -9999)
        
        # Calculate CHM by subtracting DEM from DSM
        chm_data = dsm_data - dem_data
        
        # Mask vegetation below threshold
        mask_small = chm_data < 1.83
        
        # Keep DEM value if DSM and DEM values are equal
        chm_data[dsm_data == dem_data] = dem_data[dsm_data == dem_data]
        
        # Apply masks
        chm_data[mask_null | mask_small] = 0
        
        # Update CHM profile
        chm_profile = resized_dsm.profile.copy()
        chm_profile.update(dtype=rasterio.float32, count=1)

        # Write CHM to output file
        with rasterio.open(chm_path, 'w', **chm_profile) as chm:
            chm.write(chm_data.astype(np.float32), 1)

    print(f"CHM successfully created and saved to {chm_path}")

# Step 1: Create the shapefile for the given extent
def create_shapefile(extent, crs_epsg, output_shapefile):
    """
    Create a shapefile with the given extent.

    Args:
        extent (tuple): Tuple of coordinates defining the extent (left, right, bottom, top).
        crs_epsg (int): EPSG code of the CRS.
        output_shapefile (str): Path to save the output shapefile.
    """
    # Create a box geometry for the extent
    geom = box(extent[0], extent[3], extent[1], extent[2])

    # Create a GeoDataFrame with the geometry
    gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs=f"EPSG:{crs_epsg}")

    # Save as a shapefile
    gdf.to_file(output_shapefile, engine="pyogrio")

# Step 2: Clip the raster using the shapefile
def clip_raster_with_shapefile(input_raster, shapefile, output_raster):
    """
    Clips a raster using a shapefile.

    Args:
        input_raster (str): Path to the input raster file.
        shapefile (str): Path to the shapefile to use for clipping.
        output_raster (str): Path to the output clipped raster file.
    """
    with rasterio.open(input_raster) as src:
        # Read the shapefile using pyogrio as the engine
        shapefile_gdf = gpd.read_file(shapefile, engine="pyogrio")
        geometries = [feature["geometry"] for feature in shapefile_gdf.__geo_interface__["features"]]

        # Clip the raster using the geometries
        out_image, out_transform = mask(src, geometries, crop=True)

        # Create a new profile for the output raster
        profile = src.profile
        profile.update(
            driver='GTiff',
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform
        )

        # Save the clipped raster
        with rasterio.open(output_raster, 'w', **profile) as dst:
            dst.write(out_image)