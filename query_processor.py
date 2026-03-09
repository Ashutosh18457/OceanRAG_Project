"""
MODULE 5: QUERY PROCESSOR
=========================
Accepts natural language queries and extracts:
- parameter (temperature, salinity)
- location (latitude, longitude)
- time period
Converts query into embedding.
"""

import re
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


class OceanQueryProcessor:
    """
    Processes natural language queries for oceanographic data.
    Extracts parameter, location, and time information.
    """
    
    # Parameter keywords
    PARAM_KEYWORDS = {
        'temperature': ['temperature', 'temp', 'heat', 'warming', 'cooling', 'degrees', '°C', 'celsius'],
        'salinity': ['salinity', 'salt', 'saltiness', 'salty', 'psu'],
        'pressure': ['pressure', 'depth', 'db', 'decibar'],
    }
    
    # Location patterns
    LOCATION_PATTERNS = [
        r'(\d+\.?\d*)\s*°?\s*[NnSs]\s*,?\s*(\d+\.?\d*)\s*°?\s*[EeWw]',  # 10°N, 50°E
        r'(\d+\.?\d*)\s*[NnSs]\s+(\d+\.?\d*)\s*[EeWw]',  # 10N 50E
        r'lat(?:itude)?\s*[:\-]?\s*(\-?\d+\.?\d*)',  # latitude: 10
        r'lon(?:gitude)?\s*[:\-]?\s*(\-?\d+\.?\d*)',  # longitude: 50
        r'(\-?\d+\.?\d*)\s*,\s*(\-?\d+\.?\d*)',  # 10, 50 (lat, lon)
    ]
    
    # Time patterns
    TIME_PATTERNS = [
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{4})',  # Jan 2020
        r'(\d{4})\s*[-–to]+\s*(\d{4})',  # 2020-2025
        r'year\s+(\d{4})',  # year 2020
        r'day\s+(\d+)',  # day 100
        r'month\s+(\d+)',  # month 6
    ]
    
    def __init__(self):
        """Initialize the query processor."""
        self.query = ""
        self.extracted = {
            'parameter': None,
            'location': None,
            'time_period': None,
            'raw_query': ""
        }
        
    def process(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with extracted information
        """
        self.query = query.lower().strip()
        self.extracted['raw_query'] = query
        
        # Extract components
        self.extracted['parameter'] = self._extract_parameter()
        self.extracted['location'] = self._extract_location()
        self.extracted['time_period'] = self._extract_time()
        
        return self.extracted
    
    def _extract_parameter(self) -> Optional[str]:
        """
        Extract parameter type from query.
        
        Returns:
            Parameter name or None
        """
        for param, keywords in self.PARAM_KEYWORDS.items():
            for keyword in keywords:
                if keyword in self.query:
                    return param
        return None
    
    def _extract_location(self) -> Optional[Dict[str, float]]:
        """
        Extract geographic location from query.
        
        Returns:
            Location dict with latitude and longitude
        """
        location = None
        
        # Try each pattern
        for pattern in self.LOCATION_PATTERNS:
            match = re.search(pattern, self.query, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    try:
                        lat = float(groups[0])
                        lon = float(groups[1])
                        
                        # Handle direction indicators
                        if 's' in self.query[max(0, match.start()-5):match.start()].lower():
                            lat = -abs(lat)
                        if 'w' in self.query[max(0, match.start()-5):match.start()].lower():
                            lon = -abs(lon)
                            
                        location = {'latitude': lat, 'longitude': lon}
                        break
                    except:
                        pass
        
        return location
    
    def _extract_time(self) -> Optional[Dict[str, Any]]:
        """
        Extract time period from query.
        
        Returns:
            Time period dict
        """
        time_period = {}
        
        # Check for year ranges
        year_range = re.search(r'(\d{4})\s*[-–to]+\s*(\d{4})', self.query)
        if year_range:
            time_period['start_year'] = int(year_range.group(1))
            time_period['end_year'] = int(year_range.group(2))
        
        # Check for single year
        year = re.search(r'(?:year|in|during)\s+(\d{4})', self.query)
        if year:
            time_period['year'] = int(year.group(1))
        
        # Check for month
        month = re.search(r'(?:month)\s+(\d+)', self.query)
        if month:
            time_period['month'] = int(month.group(1))
        
        # Check for day
        day = re.search(r'(?:day)\s+(\d+)', self.query)
        if day:
            time_period['day'] = int(day.group(1))
        
        # Check for month names
        month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        for i, name in enumerate(month_names):
            if name in self.query:
                time_period['month_name'] = name
                time_period['month'] = i + 1
                break
        
        return time_period if time_period else None
    
    def get_extracted_info(self) -> Dict[str, Any]:
        """
        Get all extracted information.
        
        Returns:
            Dictionary of extracted query components
        """
        return self.extracted
    
    def format_extracted(self) -> str:
        """
        Format extracted information as readable string.
        
        Returns:
            Formatted string
        """
        parts = []
        
        if self.extracted['parameter']:
            parts.append(f"Parameter: {self.extracted['parameter']}")
        
        if self.extracted['location']:
            loc = self.extracted['location']
            parts.append(f"Location: {loc['latitude']:.1f}°, {loc['longitude']:.1f}°")
        
        if self.extracted['time_period']:
            tp = self.extracted['time_period']
            time_str = []
            if 'year' in tp:
                time_str.append(f"year {tp['year']}")
            if 'start_year' in tp:
                time_str.append(f"{tp['start_year']}-{tp['end_year']}")
            if 'month' in tp:
                time_str.append(f"month {tp['month']}")
            if time_str:
                parts.append(f"Time: {', '.join(time_str)}")
        
        return " | ".join(parts) if parts else "No specific filters extracted"


class QueryToEmbedding:
    """
    Converts processed queries to embeddings.
    """
    
    def __init__(self, embedding_generator):
        """
        Initialize with embedding generator.
        
        Args:
            embedding_generator: OceanEmbeddingGenerator instance
        """
        self.embedding_generator = embedding_generator
        
    def create_query_embedding(self, query: str,
                               extracted_info: Dict[str, Any]) -> np.ndarray:
        """
        Create enriched query embedding.
        
        Args:
            query: Original query string
            extracted_info: Extracted query components
            
        Returns:
            Query embedding vector
        """
        # Build enhanced query
        enhanced_parts = [query]
        
        # Add parameter context
        if extracted_info.get('parameter'):
            param = extracted_info['parameter']
            enhanced_parts.append(f"focusing on {param} data")
        
        # Add location context
        if extracted_info.get('location'):
            loc = extracted_info['location']
            enhanced_parts.append(f"at coordinates {loc['latitude']:.1f}, {loc['longitude']:.1f}")
        
        # Add time context
        if extracted_info.get('time_period'):
            tp = extracted_info['time_period']
            if 'year' in tp:
                enhanced_parts.append(f"in year {tp['year']}")
            elif 'start_year' in tp:
                enhanced_parts.append(f"from {tp['start_year']} to {tp['end_year']}")
        
        enhanced_query = ". ".join(enhanced_parts) + "."
        
        # Generate embedding
        return self.embedding_generator.generate_embedding(enhanced_query)


def demo_queries():
    """Demo some example queries."""
    processor = OceanQueryProcessor()
    
    test_queries = [
        "What is the temperature at 10°N, 50°E?",
        "Show me salinity data in the Pacific Ocean",
        "Ocean temperature trends from 2020 to 2025",
        "Temperature anomalies at 5°N, 80°W during summer 2023",
        "Salinity variations at depth 100m near equator",
        "Compare temperature and salinity in the Atlantic"
    ]
    
    print("Query Processing Demo:")
    print("=" * 60)
    
    for query in test_queries:
        result = processor.process(query)
        print(f"\nQuery: {query}")
        print(f"Extracted: {processor.format_extracted()}")


if __name__ == "__main__":
    # Test query processing
    demo_queries()

