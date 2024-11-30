import pdal
import json
import rasterio
import numpy as np

# def create_dem(input_path, output_path):
#     # Define the PDAL pipeline to create a DEM
#     pipeline = [
#         {
#             "type": "readers.las",
#             "filename": input_path
#         },
#         {
#             "type": "filters.range",
#             "limits": "Classification[2:2]"  # Adjust classification if needed
#         },
#         {
#             "type": "filters.smrf",  # Remove noise and classify ground points
#             "window": 16,  # Size of the moving window for filtering
#             "threshold": 0.5,  # Minimum height difference for points to be classified as ground
#             "slope": 0.15  # Slope of the ground surface
#         },
#         {
#             "type": "filters.delaunay"  # Create TIN from point cloud
#         },
#         {
#             "type": "filters.outlier",  # Optional: Remove outliers
#             "method": "radius",
#             "radius": 1.0
#         },
#         {
#             "type": "filters.ferry",
#             "dimensions": "Z=>Elevation"  # Rename Z to Elevation if needed
#         },
#         {
#             "type": "writers.gdal",
#             "filename": output_path,
#             "resolution": 1.0,  # Resolution of the output DEM
#             "output_type": "max",
#             "gdaldriver": "GTiff"
#         }
#     ]
    
    
#     # Create and execute the PDAL pipeline
#     pipeline_obj = pdal.Pipeline(json.dumps(pipeline))
#     pipeline_obj.execute()

#     print(f'DEM created at {output_path}')




def create_dsm_old(input_path, output_path):
    # Define the PDAL pipeline to create a DSM
    # Define PDAL pipeline for DSM
    pipeline = [
        {
            "type": "readers.las",
            "filename": input_path
        },
        {
            "type": "filters.range",
            "limits": "returnnumber[1:1], Classification[4:5], Classification[2:2]"
        },
        {
            "type": "writers.gdal",
            "filename": output_path,
            "output_type": "max",
            "resolution": 2,
            "radius": 1,
            "gdaldriver": "GTiff"
        }
    ]


    # Create and execute the PDAL pipeline
    pipeline_obj = pdal.Pipeline(json.dumps(pipeline))
    pipeline_obj.execute()

    print(f'DSM created at {output_path}')


def create_dsm(input_path, output_path):
    # Get metadata of the input LAS file to extract CRS
    pipeline_metadata = [
        {
            "type": "readers.las",
            "filename": input_path
        }
    ]

    # Create a pipeline to fetch metadata
    pipeline_obj_metadata = pdal.Pipeline(json.dumps(pipeline_metadata))
    pipeline_obj_metadata.execute()
    
    # Extract CRS (spatial reference) from the metadata
    metadata = pipeline_obj_metadata.metadata
    # print(metadata)
    crs = metadata['metadata']['readers.las']['spatialreference']
    # Now, define the pipeline to create a DSM using the same CRS as the input
    pipeline = [
        {
            "type": "readers.las",
            "filename": input_path
        },
        {
            "type": "filters.range",
            "limits": "returnnumber[1:1], Classification[4:5], Classification[2:2]"
        },
        {
            "type": "writers.gdal",
            "filename": output_path,
            "output_type": "max",
            "resolution": 2,
            "radius": 1,
            "gdaldriver": "GTiff",
            "spatialreference": crs  # Apply the extracted CRS to the output DSM
        }
    ]

    # Create and execute the PDAL pipeline
    pipeline_obj = pdal.Pipeline(json.dumps(pipeline))
    pipeline_obj.execute()

    print(f'DSM created at {output_path}')


# def create_chm(dsm_path, dem_path, chm_path):
#     # Read DSM
#     with rasterio.open(dsm_path) as dsm_src:
#         dsm = dsm_src.read(1)
#         dsm_meta = dsm_src.meta
#         dsm_nodata = dsm_src.nodata

#     # Read DEM
#     with rasterio.open(dem_path) as dem_src:
#         dem = dem_src.read(1)
#         dem_nodata = dem_src.nodata

#     # Ensure DEM and DSM have the same shape
#     if dsm.shape != dem.shape:
#         raise ValueError("DSM and DEM must have the same dimensions.")

#     # Handle NoData values
#     if dsm_nodata is not None:
#         dsm = np.where(dsm == dsm_nodata, np.nan, dsm)
#     if dem_nodata is not None:
#         dem = np.where(dem == dem_nodata, np.nan, dem)

#     # Calculate CHM
#     chm = np.where(np.isnan(dsm) | np.isnan(dem), np.nan, dsm - dem)

#     # Update metadata for CHM
#     chm_meta = dsm_meta.copy()
#     chm_meta.update({
#         'count': 1,
#         'nodata': 0  # Set a new NoData value for CHM
#     })

#     # Write CHM to file
#     with rasterio.open(chm_path, 'w', **chm_meta) as chm_dst:
#         # Convert NaNs to NoData value
#         chm = np.where(np.isnan(chm), chm_meta['nodata'], chm)
#         chm_dst.write(chm, 1)

#     print(f'CHM created at {chm_path}')