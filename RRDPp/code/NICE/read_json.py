import json
import glob

files = glob.glob('/dmidata/users/ilo/projects/RRDPp/RawData/NICE/Buoys/IMB_2015a.json')

# Load the JSON file
with open(files[0], "r") as file:
    data = json.load(file)

# Check that it’s a Feature with LineString geometry
if data.get("type") == "Feature" and data.get("geometry", {}).get("type") == "LineString":
    coordinates = data["geometry"]["coordinates"]
    
    # Now you can work with the coordinates
    print("Number of coordinate points:", len(coordinates))
    print("First 5 coordinate points:")
    for coord in coordinates[:5]:
        print(coord)