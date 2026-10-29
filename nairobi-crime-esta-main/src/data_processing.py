"""
Data processing and generation utilities for the Nairobi Crime ESTA project.
"""
import pandas as pd
import numpy as np

def generate_mock_nairobi_data(seed: int = 42) -> pd.DataFrame:
    """
    Generates a mock dataset of crime incidents across the 17 sub-counties of Nairobi.
    
    Args:
        seed (int): Random seed for reproducibility. Default is 42.
        
    Returns:
        pd.DataFrame: A dataframe containing sub_county, population, crime_incidents, latitude, and longitude.
    """
    sub_counties = [
        'Westlands', 'Dagoretti North', 'Dagoretti South', 'Langata', 'Kibra',
        'Roysambu', 'Kasarani', 'Ruaraka', 'Embakasi South', 'Embakasi North',
        'Embakasi Central', 'Embakasi East', 'Embakasi West', 'Makadara',
        'Kamukunji', 'Starehe', 'Mathare'
    ]
    
    np.random.seed(seed)
    
    mock_data = pd.DataFrame({
        'sub_county': sub_counties,
        'population': np.random.randint(100000, 450000, size=17),
        'crime_incidents': np.random.randint(50, 600, size=17),
        'latitude': np.random.uniform(-1.36, -1.20, size=17),
        'longitude': np.random.uniform(36.68, 36.96, size=17)
    })
    
    return mock_data