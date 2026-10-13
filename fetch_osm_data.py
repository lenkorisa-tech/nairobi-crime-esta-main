import osmnx as ox
import os
import warnings
warnings.filterwarnings('ignore')

# Helper function to fix the "Invalid field type <class 'list'>" error
def clean_gdf_for_geojson(gdf):
    """Converts any list or dict columns to strings to prevent Fiona GeoJSON errors."""
    for col in gdf.columns:
        # Check if the column contains lists or dicts and convert them to text
        if gdf[col].apply(lambda x: isinstance(x, (list, dict))).any():
            gdf[col] = gdf[col].astype(str)
    return gdf

# Define the study area
place_name = "Nairobi City County, Kenya"
output_dir = "data/raw/osm" 
os.makedirs(output_dir, exist_ok=True)

print(f"--- Extracting OSM Data for {place_name} ---")

# 1. Road Network Density (Proxy for Crime Pattern Theory "edges")
print("Downloading road network... (This may take 2-3 minutes)")
try:
    G = ox.graph_from_place(place_name, network_type="drive")
    nodes, edges = ox.graph_to_gdfs(G)
    edges = clean_gdf_for_geojson(edges) # Clean the data before saving
    edges.to_file(f"{output_dir}/nairobi_road_network.geojson", driver="GeoJSON")
    print("✅ Road network saved.")
except Exception as e:
    print(f"⚠️ Road network download failed: {e}")

# 2. Points of Interest (Proxy for Routine Activity Theory "nodes/targets")
print("Downloading Points of Interest (POIs)...")
try:
    tags = {
        "amenity": ["police", "bank", "atm", "school", "hospital", "marketplace"],
        "shop": ["supermarket", "mall"],
        "tourism": ["hotel"]
    }
    pois = ox.features_from_place(place_name, tags=tags)
    pois = clean_gdf_for_geojson(pois) # Clean the data before saving
    pois.to_file(f"{output_dir}/nairobi_pois.geojson", driver="GeoJSON")
    print("✅ POIs saved.")
except Exception as e:
    print(f"⚠️ POI download failed: {e}")

# 3. Land Use (Proxy for "suitable targets" and "capable guardians")
print("Downloading Land Use polygons...")
try:
    landuse_tags = {"landuse": True, "amenity": True}
    landuse = ox.features_from_place(place_name, tags=landuse_tags)
    landuse = clean_gdf_for_geojson(landuse) # Clean the data before saving
    landuse.to_file(f"{output_dir}/nairobi_landuse.geojson", driver="GeoJSON")
    print("✅ Land use saved.")
except Exception as e:
    print(f"⚠️ Land use download failed: {e}")

print("\n🎉 OSM data extraction complete! Check data/raw/osm/")
