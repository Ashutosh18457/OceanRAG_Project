"""
MODULE 2: DATA PREPROCESSING
============================
Converts oceanographic datasets into structured records and 
splits them into retrieval-friendly chunks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import json


class OceanDataPreprocessor:
    """
    Preprocesses oceanographic data for retrieval.
    Converts multidimensional arrays into structured text records.
    """
    
    def __init__(self, variables: Dict[str, np.ndarray], 
                 coordinates: Dict[str, np.ndarray]):
        """
        Initialize preprocessor with loaded data.
        
        Args:
            variables: Dictionary of oceanographic variables
            coordinates: Dictionary of coordinate arrays
        """
        self.variables = variables
        self.coordinates = coordinates
        self.records = []
        self.chunks = []
        
    def create_text_description(self, param: str, value: float, 
                                 lat: float, lon: float, 
                                 depth: Optional[float] = None,
                                 time: Optional[Any] = None) -> str:
        """
        Create a natural language text description of an ocean data point.
        
        Args:
            param: Parameter name (temperature, salinity, etc.)
            value: Parameter value
            lat: Latitude
            lon: Longitude
            depth: Depth (optional)
            time: Time value (optional)
            
        Returns:
            Text description string
        """
        # Format location description
        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"
        
        location = f"{abs(lat):.1f}°{lat_dir}, {abs(lon):.1f}°{lon_dir}"
        
        # Build description based on available data
        parts = []
        
        # Parameter description
        param_descriptions = {
            'temperature': 'sea surface temperature',
            'salinity': 'ocean salinity', 
            'pressure': 'water pressure',
            'temperature': 'water temperature'
        }
        param_desc = param_descriptions.get(param, param)
        parts.append(f"The {param_desc} is {value:.2f}")
        
        # Add location
        parts.append(f"at location {location}")
        
        # Add depth if available
        if depth is not None:
            parts.append(f"at depth {depth:.1f} meters")
        
        # Add time context
        if time is not None:
            if isinstance(time, (int, float)):
                parts.append(f"on day {time}")
            else:
                parts.append(f"at time {time}")
        
        return ". ".join(parts) + "."
    
    def generate_records(self, max_points: int = 10000) -> List[Dict[str, Any]]:
        """
        Convert the dataset into structured records.
        
        Args:
            max_points: Maximum number of records to generate
            
        Returns:
            List of structured records
        """
        records = []
        
        # Get variable shapes
        temp = self.variables.get('temperature')
        salt = self.variables.get('salinity')
        lat = self.coordinates.get('latitude')
        lon = self.coordinates.get('longitude')
        
        if lat is None or lon is None:
            raise ValueError("Latitude and longitude coordinates required")
        
        # Handle different array shapes
        lat_1d = np.atleast_1d(lat)
        lon_1d = np.atleast_1d(lon)
        
        # Generate records based on available data dimensions
        if temp is not None:
            temp_shape = temp.shape
            n_times = temp_shape[0] if len(temp_shape) > 0 else 1
            n_depths = temp_shape[1] if len(temp_shape) > 1 else 1
            n_lats = temp_shape[2] if len(temp_shape) > 2 else len(lat_1d)
            n_lons = temp_shape[3] if len(temp_shape) > 3 else len(lon_1d)
            
            # Sample points if too many
            total_points = n_times * n_depths * n_lats * n_lons
            if total_points > max_points:
                print(f"⚠ Sampling {max_points} from {total_points} points")
                indices = np.random.choice(total_points, max_points, replace=False)
            else:
                indices = range(total_points)
            
            time_var = self.coordinates.get('time')
            depth_var = self.coordinates.get('depth')
            
            for idx in indices:
                # Calculate multidimensional indices
                if total_points > max_points:
                    idx = int(idx)
                
                t = (idx // (n_depths * n_lats * n_lons)) % n_times
                d = (idx // (n_lats * n_lons)) % n_depths
                la = (idx // n_lons) % n_lats
                lo = idx % n_lons
                
                lat_val = lat_1d[la] if la < len(lat_1d) else lat_1d[-1]
                lon_val = lon_1d[lo] if lo < len(lon_1d) else lon_1d[-1]
                temp_val = temp[t, d, la, lo] if len(temp_shape) > 3 else temp[t, d]
                
                time_val = time_var[t] if time_var is not None and t < len(time_var) else t
                depth_val = depth_var[d] if depth_var is not None and d < len(depth_var) else d * 20
                
                # Create record
                record = {
                    'parameter': 'temperature',
                    'value': float(temp_val),
                    'latitude': float(lat_val),
                    'longitude': float(lon_val),
                    'depth': float(depth_val),
                    'time': float(time_val) if isinstance(time_val, (int, float, np.integer, np.floating)) else str(time_val),
                    'text': self.create_text_description('temperature', temp_val, lat_val, lon_val, depth_val, time_val)
                }
                records.append(record)
        
        # Add salinity records if available
        if salt is not None:
            salt_shape = salt.shape
            n_times = salt_shape[0] if len(salt_shape) > 0 else 1
            n_depths = salt_shape[1] if len(salt_shape) > 1 else 1
            n_lats = salt_shape[2] if len(salt_shape) > 2 else len(lat_1d)
            n_lons = salt_shape[3] if len(salt_shape) > 3 else len(lon_1d)
            
            time_var = self.coordinates.get('time')
            depth_var = self.coordinates.get('depth')
            
            for t in range(min(n_times, 10)):  # Limit salinity points
                for d in range(min(n_depths, 5)):
                    for la in range(min(n_lats, 5)):
                        for lo in range(min(n_lons, 5)):
                            lat_val = lat_1d[la]
                            lon_val = lon_1d[lo]
                            salt_val = salt[t, d, la, lo] if len(salt_shape) > 3 else salt[t, d]
                            
                            time_val = time_var[t] if time_var is not None else t
                            depth_val = depth_var[d] if depth_var is not None else d * 20
                            
                            record = {
                                'parameter': 'salinity',
                                'value': float(salt_val),
                                'latitude': float(lat_val),
                                'longitude': float(lon_val),
                                'depth': float(depth_val),
                                'time': float(time_val) if isinstance(time_val, (int, float, np.integer, np.floating)) else str(time_val),
                                'text': self.create_text_description('salinity', salt_val, lat_val, lon_val, depth_val, time_val)
                            }
                            records.append(record)
        
        self.records = records
        print(f"✓ Generated {len(records)} records")
        return records
    
    def chunk_records(self, chunk_size: int = 100) -> List[Dict[str, Any]]:
        """
        Split records into chunks for efficient retrieval.
        
        Args:
            chunk_size: Number of records per chunk
            
        Returns:
            List of chunked records
        """
        if not self.records:
            raise ValueError("No records generated. Call generate_records() first.")
        
        chunks = []
        for i in range(0, len(self.records), chunk_size):
            chunk_records = self.records[i:i + chunk_size]
            
            # Aggregate chunk statistics
            values = [r['value'] for r in chunk_records]
            lats = [r['latitude'] for r in chunk_records]
            lons = [r['longitude'] for r in chunk_records]
            
            param = chunk_records[0]['parameter']
            
            chunk = {
                'chunk_id': i // chunk_size,
                'parameter': param,
                'count': len(chunk_records),
                'mean_value': float(np.mean(values)),
                'std_value': float(np.std(values)),
                'min_value': float(np.min(values)),
                'max_value': float(np.max(values)),
                'latitude_range': [float(np.min(lats)), float(np.max(lats))],
                'longitude_range': [float(np.min(lons)), float(np.max(lons))],
                'records': chunk_records,
                'text_summary': f"Ocean {param} data: mean={np.mean(values):.2f}, "
                              f"std={np.std(values):.2f}, range=[{np.min(values):.2f}, {np.max(values):.2f}]"
            }
            chunks.append(chunk)
        
        self.chunks = chunks
        print(f"✓ Created {len(chunks)} chunks of size {chunk_size}")
        return chunks
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert records to pandas DataFrame.
        
        Returns:
            DataFrame of records
        """
        return pd.DataFrame(self.records)
    
    def get_parameter_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each parameter.
        
        Returns:
            Dictionary of parameter statistics
        """
        stats = {}
        
        for record in self.records:
            param = record['parameter']
            if param not in stats:
                stats[param] = {'values': [], 'count': 0}
            stats[param]['values'].append(record['value'])
            stats[param]['count'] += 1
        
        for param, data in stats.items():
            values = np.array(data['values'])
            stats[param] = {
                'count': data['count'],
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return stats


if __name__ == "__main__":
    # Test preprocessing
    print("Testing OceanDataPreprocessor...")
    
    # Create sample data
    from data_loader import create_sample_data, OceanDataLoader
    
    create_sample_data("test_data.nc")
    loader = OceanDataLoader("test_data.nc")
    loader.load_dataset()
    variables = loader.extract_variables()
    coordinates = loader.get_coordinates()
    
    # Test preprocessing
    preprocessor = OceanDataPreprocessor(variables, coordinates)
    records = preprocessor.generate_records(max_points=100)
    chunks = preprocessor.chunk_records(chunk_size=20)
    
    print(f"\nParameter Statistics:")
    stats = preprocessor.get_parameter_stats()
    for param, stat in stats.items():
        print(f"  {param}: {stat}")
    
    loader.close()

