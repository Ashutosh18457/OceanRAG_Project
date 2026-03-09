"""
MODULE 1: DATA LOADER
=====================
This module handles loading oceanographic data from NETCDF files.
Extracts: temperature, salinity, pressure, latitude, longitude, time
"""

import xarray as xr
import netCDF4 as nc
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import warnings
import os
import zipfile

warnings.filterwarnings('ignore')


class OceanDataLoader:
    """
    Loads and parses oceanographic data from NETCDF format files.
    Supports standard oceanographic datasets with temperature, salinity,
    pressure, and coordinate variables.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the data loader with path to NETCDF file.
        
        Args:
            file_path: Path to the NETCDF file
        """
        self.file_path = file_path
        self.dataset = None
        self.variables = {}
        self.coordinates = {}
        
    def load_dataset(self) -> xr.Dataset:
        """
        Load the NETCDF dataset using xarray.
        
        Returns:
            xarray.Dataset: Loaded oceanographic dataset
        """
        try:
            # Try loading with xarray (supports multiple NETCDF formats)
            self.dataset = xr.open_dataset(self.file_path, engine='netcdf4')
            print(f"Successfully loaded dataset: {self.file_path}")
            print(f"  Dimensions: {list(self.dataset.dims.keys())}")
            print(f"  Variables: {list(self.dataset.data_vars)}")
            return self.dataset
        except Exception as e:
            print(f"Error loading with xarray: {e}")
            # Fallback to netCDF4
            try:
                self.dataset = nc.Dataset(self.file_path, 'r')
                print(f"Successfully loaded with netCDF4: {self.file_path}")
                return self.dataset
            except Exception as e2:
                raise ValueError(f"Failed to load NETCDF file: {e2}")
    
    def extract_variables(self) -> Dict[str, np.ndarray]:
        """
        Extract key oceanographic variables from the dataset.
        
        Returns:
            Dictionary containing extracted variables
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Standard oceanographic variable names to look for
        var_mappings = {
            'temperature': ['temperature', 'temp', 'sst', 'sea_water_temperature', 'T'],
            'salinity': ['salinity', 'salt', 'sss', 'sea_water_salinity', 'S'],
            'pressure': ['pressure', 'pres', 'depth', 'sea_water_pressure', 'P'],
            'latitude': ['latitude', 'lat', 'y'],
            'longitude': ['longitude', 'lon', 'x'],
            'time': ['time', 't', 'ocean_time']
        }
        
        extracted = {}
        
        # Try to find and extract each variable
        for var_name, possible_names in var_mappings.items():
            for possible_name in possible_names:
                if possible_name in self.dataset.data_vars or possible_name in self.dataset.variables:
                    if hasattr(self.dataset, possible_name):
                        data = self.dataset[possible_name]
                        if hasattr(data, 'values'):
                            extracted[var_name] = data.values
                        else:
                            extracted[var_name] = np.array(data)
                        print(f"Extracted {var_name}: {extracted[var_name].shape}")
                        break
                    elif possible_name in self.dataset.variables:
                        extracted[var_name] = np.array(self.dataset.variables[possible_name][:])
                        print(f"Extracted {var_name}: {extracted[var_name].shape}")
                        break
        
        self.variables = extracted
        return extracted
    
    def get_coordinates(self) -> Dict[str, np.ndarray]:
        """
        Extract coordinate variables (lat, lon, time, depth).
        
        Returns:
            Dictionary of coordinate arrays
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded.")
        
        coords = {}
        
        # Extract latitude
        for lat_name in ['latitude', 'lat', 'y']:
            if lat_name in self.dataset.coords:
                coords['latitude'] = self.dataset.coords[lat_name].values
                break
        
        # Extract longitude  
        for lon_name in ['longitude', 'lon', 'x']:
            if lon_name in self.dataset.coords:
                coords['longitude'] = self.dataset.coords[lon_name].values
                break
        
        # Extract time
        for time_name in ['time', 't']:
            if time_name in self.dataset.coords:
                coords['time'] = self.dataset.coords[time_name].values
                break
        
        # Extract depth/pressure
        for depth_name in ['depth', 'pressure', 'pres', 'z']:
            if depth_name in self.dataset.coords:
                coords['depth'] = self.dataset.coords[depth_name].values
                break
        
        self.coordinates = coords
        print(f"Extracted coordinates: {list(coords.keys())}")
        return coords
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the loaded data.
        
        Returns:
            Dictionary with data summary
        """
        summary = {
            'file_path': self.file_path,
            'variables': list(self.variables.keys()),
            'coordinates': list(self.coordinates.keys()),
            'shapes': {}
        }
        
        for var_name, data in self.variables.items():
            summary['shapes'][var_name] = data.shape
            if hasattr(data, 'shape') and data.size > 0:
                summary[f'{var_name}_min'] = float(np.nanmin(data))
                summary[f'{var_name}_max'] = float(np.nanmax(data))
                summary[f'{var_name}_mean'] = float(np.nanmean(data))
        
        return summary
    
    def close(self):
        """Close the dataset."""
        if self.dataset is not None:
            if hasattr(self.dataset, 'close'):
                self.dataset.close()
            print("Dataset closed")


def create_sample_data(output_path: str = "sample_ocean_data.nc"):
    """
    Create sample oceanographic data for testing.
    
    Args:
        output_path: Path to save the sample NETCDF file
    """
    import numpy as np
    from netCDF4 import Dataset
    
    # Create dimensions
    time_size = 365  # 1 year of daily data
    depth_size = 50
    lat_size = 10
    lon_size = 10
    
    with Dataset(output_path, 'w', format='NETCDF4') as ncfile:
        # Create dimensions
        ncfile.createDimension('time', time_size)
        ncfile.createDimension('depth', depth_size)
        ncfile.createDimension('latitude', lat_size)
        ncfile.createDimension('longitude', lon_size)
        
        # Create variables
        times = ncfile.createVariable('time', 'f8', ('time',))
        depths = ncfile.createVariable('depth', 'f8', ('depth',))
        lats = ncfile.createVariable('latitude', 'f8', ('latitude',))
        lons = ncfile.createVariable('longitude', 'f8', ('longitude',))
        
        # Fill coordinate variables
        times[:] = np.arange(time_size)
        depths[:] = np.linspace(0, 1000, depth_size)  # 0-1000m depth
        lats[:] = np.linspace(-45, 45, lat_size)
        lons[:] = np.linspace(-90, 90, lon_times := lon_size)
        
        # Create data variables with realistic ocean values
        temp = ncfile.createVariable('temperature', 'f8', ('time', 'depth', 'latitude', 'longitude'))
        salt = ncfile.createVariable('salinity', 'f8', ('time', 'depth', 'latitude', 'longitude'))
        pres = ncfile.createVariable('pressure', 'f8', ('depth',))
        
        # Generate realistic ocean data
        pres[:] = depths[:] * 10  # Pressure in db (approximately depth * 10)
        
        for t in range(time_size):
            for d in range(depth_size):
                for la in range(lat_size):
                    for lo in range(lon_size):
                        # Temperature: decreases with depth, varies with latitude
                        base_temp = 25 - (lats[la] / 45) * 10  # Warmer at equator
                        depth_factor = np.exp(-d / 20)  # Decreases exponentially with depth
                        seasonal = 3 * np.sin(2 * np.pi * t / 365)  # Seasonal variation
                        temp[t, d, la, lo] = base_temp * depth_factor + seasonal + np.random.randn() * 0.5
                        
                        # Salinity: varies with depth and latitude
                        base_salt = 35 - abs(lats[la]) / 45 * 2  # Higher at poles
                        salt[t, d, la, lo] = base_salt + d / depth_size * 0.5 + np.random.randn() * 0.2
        
        # Add units
        times.units = 'days since 2020-01-01'
        depths.units = 'm'
        lats.units = 'degrees_north'
        lons.units = 'degrees_east'
        temp.units = 'degrees_Celsius'
        salt.units = 'PSU'
        pres.units = 'db'
        
        print(f"Created sample data: {output_path}")


class KaggleDatasetLoader:
    """
    Downloads and loads oceanographic datasets from Kaggle.
    Supports authentication via Kaggle API credentials.
    """
    
    def __init__(self, dataset_name: str, download_path: str = "./data"):
        """
        Initialize the Kaggle dataset loader.
        
        Args:
            dataset_name: Kaggle dataset name in format 'owner/dataset-name'
                        e.g., 'sohier/oceansat-iss-ocean-color'
            download_path: Local directory to download the dataset
        """
        self.dataset_name = dataset_name
        self.download_path = download_path
        self.dataset = None
        self.downloaded_file = None
        
    def authenticate(self, username: str = None, key: str = None) -> bool:
        """
        Authenticate with Kaggle API.
        
        Args:
            username: Kaggle username (optional if credentials file exists)
            key: Kaggle API key (optional if credentials file exists)
            
        Returns:
            bool: True if authentication successful
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()
            
            print("Successfully authenticated with Kaggle")
            return True
            
        except ImportError:
            print("Kaggle package not installed. Install with: pip install kaggle")
            return False
        except Exception as e:
            print(f"Authentication failed: {e}")
            print("  To use Kaggle API:")
            print("  1. Go to https://www.kaggle.com/account")
            print("  2. Click 'Create New API Token'")
            print("  3. Place 'kaggle.json' in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
            return False
    
    def download_dataset(self, overwrite: bool = False) -> str:
        """
        Download dataset from Kaggle.
        
        Args:
            overwrite: Whether to overwrite existing files
            
        Returns:
            str: Path to downloaded file
            
        Raises:
            Exception: If download fails
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            # Ensure download directory exists
            os.makedirs(self.download_path, exist_ok=True)
            
            # Authenticate
            api = KaggleApi()
            api.authenticate()
            
            print(f"Downloading dataset: {self.dataset_name}")
            
            # Download dataset
            api.dataset_download_files(
                self.dataset_name,
                path=self.download_path,
                unzip=not overwrite  # Unzip unless overwriting
            )
            
            # Find the downloaded file
            downloaded_files = []
            for f in os.listdir(self.download_path):
                if f.endswith('.nc') or f.endswith('.csv') or f.endswith('.zip'):
                    full_path = os.path.join(self.download_path, f)
                    downloaded_files.append(full_path)
                    print(f"Downloaded: {f}")
            
            if downloaded_files:
                self.downloaded_file = downloaded_files[0]
                return self.downloaded_file
            else:
                raise ValueError("No data files found in downloaded dataset")
                
        except Exception as e:
            raise ValueError(f"Failed to download dataset: {e}")
    
    def load_with_ocean_loader(self) -> 'OceanDataLoader':
        """
        Load the downloaded dataset using OceanDataLoader.
        
        Returns:
            OceanDataLoader: Configured data loader instance
            
        Raises:
            ValueError: If no dataset has been downloaded
        """
        if self.downloaded_file is None:
            # Try to download first
            self.download_dataset()
        
        # Check if file exists
        if not os.path.exists(self.downloaded_file):
            raise ValueError(f"Downloaded file not found: {self.downloaded_file}")
        
        # Handle zip files
        if self.downloaded_file.endswith('.zip'):
            with zipfile.ZipFile(self.downloaded_file, 'r') as zip_ref:
                # Extract to same directory
                extract_path = self.download_path
                zip_ref.extractall(extract_path)
                print(f"Extracted zip file to: {extract_path}")
                
                # Find the extracted netCDF file
                for f in os.listdir(extract_path):
                    if f.endswith('.nc'):
                        self.downloaded_file = os.path.join(extract_path, f)
                        break
        
        # Load using OceanDataLoader
        loader = OceanDataLoader(self.downloaded_file)
        loader.load_dataset()
        
        print(f"Loaded Kaggle dataset: {self.downloaded_file}")
        return loader
    
    @staticmethod
    def list_ocean_datasets() -> list:
        """
        List popular oceanographic datasets on Kaggle.
        
        Returns:
            list: List of popular ocean datasets
        """
        popular_datasets = [
            {"name": "sohier/oceansat-iss-ocean-color", "description": "Ocean color data from ISS"},
            {"name": "adityajn105/ocean-surface-sediment", "description": "Ocean surface sediment data"},
            {"name": "kennethjiang/sea-surface-temperature", "description": "Sea surface temperature"},
            {"name": "NOAA-GFDL/gfdl-ocean-model", "description": "GFDL Ocean Model data"},
            {"name": "ucsb-vrl/bisqueucsb/oceanographic-dataset", "description": "Oceanographic measurements"},
            {"name": "ashutoshbhandari/ocean-temperature", "description": "Ocean temperature data"},
            {"name": "santoshsunny/ocean-salinity", "description": "Ocean salinity measurements"},
        ]
        
        print("Popular Oceanographic Datasets on Kaggle:")
        for i, ds in enumerate(popular_datasets, 1):
            print(f"  {i}. {ds['name']}")
            print(f"     {ds['description']}")
        
        return popular_datasets


def load_from_kaggle(dataset_name: str, download_path: str = "./data") -> 'OceanDataLoader':
    """
    Convenience function to download and load a Kaggle dataset.
    
    Args:
        dataset_name: Kaggle dataset name in format 'owner/dataset-name'
        download_path: Local directory to download the dataset
        
    Returns:
        OceanDataLoader: Ready-to-use data loader instance
    """
    kaggle_loader = KaggleDatasetLoader(dataset_name, download_path)
    
    # Try to authenticate
    if not kaggle_loader.authenticate():
        print("Kaggle authentication required for downloading datasets")
        return None
    
    # Download and load
    ocean_loader = kaggle_loader.load_with_ocean_loader()
    return ocean_loader


if __name__ == "__main__":
    # Test the data loader
    print("Testing OceanDataLoader...")
    
    # Create sample data
    create_sample_data("sample_ocean_data.nc")
    
    # Load and test
    loader = OceanDataLoader("sample_ocean_data.nc")
    loader.load_dataset()
    loader.extract_variables()
    loader.get_coordinates()
    summary = loader.get_data_summary()
    print("\nData Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    loader.close()

