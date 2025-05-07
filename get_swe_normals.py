import datetime as dt
import os

import numpy as np
import pandas as pd

import xarray as xr
import zarr
import rioxarray as rio
from azure.storage.blob import ContainerClient, generate_container_sas, ContainerSasPermissions

from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

SNODAS_SAS = os.getenv("SNODAS_SAS")
COPERNICUS_SAS = os.getenv("COPERNICUS_SAS")

def get_dataset(container_name, zarr_prefix, sas):
    """
    open a zarr dataset from blob storage

    container_name: str
    zarr_prefix: str

    returns: xarray dataset
    """
    url = "https://climatedataprod.blob.core.windows.net?" + sas
    container_client = ContainerClient(url, container_name=container_name)
    store = zarr.ABSStore(client=container_client, prefix=zarr_prefix)
    ds = xr.open_zarr(store)
    return ds

def get_month_year():
    today = dt.datetime.today()
    default_month = today.month
    default_year = today.year

    year_input = input(f"Enter year [default: {default_year}]: ") or str(default_year)
    try:
        year = int(year_input)
        if year < 1900 or year > 2100:
            raise ValueError
    except ValueError:
        print("Invalid year. Please enter a 4-digit number.")
        return None, None

    month_input = input(f"Enter month (1-12) [default: {default_month}]: ") or str(default_month)
    try:
        month = int(month_input)
        if month < 1 or month > 12:
            raise ValueError
    except ValueError:
        print("Invalid month. Please enter a number from 1 to 12.")
        return None, None

    return year, month

def get_normals(da, year, month, snodas=True):
    # Generate the list of first-of-month dates (at 5am) up to the year before the input year
    if snodas:
        month_firsts = pd.date_range(
            dt.datetime(2008, month, 1, 5),
            dt.datetime(year - 1, month, 1, 5),
            freq="AS"  # annual on start of specified month
        )
    else:
        month_firsts = pd.date_range(
            dt.datetime(2000, month, 1, 0),
            dt.datetime(year - 1, month, 1, 0),
            freq="AS"  # annual on start of specified month
        )

    # Filter the DataArray to only include those dates
    month_firsts_da = da.sel(time=da.time.isin(month_firsts))

    # Compute normals
    normals = month_firsts_da.mean(dim='time')
    return normals

def get_percent_of_normal(da, normals, year, month, snodas=True):
    # Select the same month in the input year
    if snodas:
        first_of_month = dt.datetime(year, month, 1, 5)
    else:
        first_of_month = dt.datetime(year, month, 1, 0)

    da_this_year_first_of_month = da.sel(time=first_of_month)

    # Calculate percent of normal
    percent_of_normal = (da_this_year_first_of_month / normals) * 100
    return percent_of_normal


def main():
    year, month = get_month_year()
    if year is None or month is None:
        print("Exiting due to invalid input.")
        return

    # Import the snodas dataset
    snodas_ds = get_dataset('snodas-v1', 'snodas.zarr', SNODAS_SAS)
    print('snodas dataset loaded')

    # Get a spatial subset of the snodas dataset
    snodas_da = snodas_ds['1034'].sel(
        x=slice(-14528544.60, -10528544.60),
        y=slice(5915989.877651, 7255989.877651)
        )

    # import the copernicus dataset
    copernicus_ds = get_dataset('copernicus', 'copernicus.zarr', COPERNICUS_SAS)
    print('copernicus dataset loaded')

    cop_da = copernicus_ds['swe'].sel(
        x=slice(-15029545, -10929545),
        y=slice(5963905, 8263905)
        )

    # Get normals for the specified month and year
    snodas_normals = get_normals(snodas_da, year, month, snodas=True)

    copernicus_normals = get_normals(cop_da, year, month, snodas=False)

    print('processing percent of normal')
    snodas_percent_of_normal = get_percent_of_normal(snodas_da, 
                                                     snodas_normals, 
                                                     year, 
                                                     month, 
                                                     snodas=True)
    
    copernicus_percent_of_normal = get_percent_of_normal(cop_da,
                                                        copernicus_normals, 
                                                        year, 
                                                        month, 
                                                        snodas=False)

    snodas_percent_of_normal = snodas_percent_of_normal.rio.write_crs("EPSG:3857")
    copernicus_percent_of_normal = copernicus_percent_of_normal.rio.write_crs("EPSG:3857")

    # Export with dynamic filename
    month_str = dt.date(year, month, 1).strftime("%b").lower()

    snodas_output_filename = f"snodas_prcnt_of_norm_{month_str}_{year}.tif"
    snodas_percent_of_normal.rio.to_raster(snodas_output_filename)
    print(f"Saved percent-of-normal raster to: {snodas_output_filename}")

    copernicus_output_filename = f"copernicus_prcnt_of_norm_{month_str}_{year}.tif"
    copernicus_percent_of_normal.rio.to_raster(copernicus_output_filename)
    print(f"Saved percent-of-normal raster to: {copernicus_output_filename}")

if __name__ == "__main__":
    main()


