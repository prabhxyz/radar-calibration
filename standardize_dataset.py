import csv
import numpy as np
from math import radians, cos, sin, atan2, sqrt

# Function to convert latitude and longitude to ECEF coordinates
def geodetic_to_ecef(lat, lon, alt):
    a = 6378137.0  # Earth radius in meters
    f = 1 / 298.257223563  # Earth flattening
    e2 = 2*f - f**2  # Square of eccentricity

    lat_rad = radians(lat)
    lon_rad = radians(lon)

    N = a / sqrt(1 - e2 * sin(lat_rad)**2)

    x = (N + alt) * cos(lat_rad) * cos(lon_rad)
    y = (N + alt) * cos(lat_rad) * sin(lon_rad)
    z = (N * (1 - e2) + alt) * sin(lat_rad)

    return x, y, z

# Function to convert AER to Cartesian coordinates
def aer_to_cartesian(azimuth, elevation, range):
    az_rad = radians(azimuth)
    el_rad = radians(elevation)

    x = range * cos(el_rad) * sin(az_rad)
    y = range * cos(el_rad) * cos(az_rad)
    z = range * sin(el_rad)

    return x, y, z

# Function to convert Cartesian coordinates to AER
def cartesian_to_aer(x, y, z):
    range_ = sqrt(x**2 + y**2 + z**2)
    elevation = atan2(z, sqrt(x**2 + y**2))
    azimuth = atan2(x, y)
    return np.degrees(azimuth), np.degrees(elevation), range_

# Function to standardize AER data based on ground station location
def standardize_aer(aer_data, station_lat, station_lon):
    station_x, station_y, station_z = geodetic_to_ecef(station_lat, station_lon, 0)
    standardized_data = []
    for azimuth, elevation, range_ in aer_data:
        x, y, z = aer_to_cartesian(azimuth, elevation, range_)
        # Translate Cartesian coordinates to station's location
        x -= station_x
        y -= station_y
        z -= station_z
        # Convert back to AER
        standardized_data.append(cartesian_to_aer(x, y, z))
    return standardized_data

# Function to read the input CSV file and write the output CSV file
def process_aer_data(input_file, output_file):
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        fieldnames_standardized = fieldnames[1:]  # Remove '_std' suffix
        fieldnames_all = [fieldnames[0]] + fieldnames_standardized

        with open(output_file, 'w', newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames_all)
            writer.writeheader()

            for row in reader:
                # Define ground station coordinates
                station_coordinates = [
                    (39.2768, -104.807),  # Ground Station 1
                    (34.583, -120.561),   # Ground Station 2
                    (33.8131, -106.659),  # Ground Station 3
                    (35.891, -110.676)    # Ground Station 4 (assuming sea level)
                ]

                # Standardize AER data for each ground station
                standardized_data = []
                for key in row.keys():
                    if key.startswith('azimuth'):
                        i = int(key.split('_')[-1])
                        aer_data = [float(row[f'azimuth_{i}']), float(row[f'elevation_{i}']), float(row[f'range_{i}'])]
                        lat, lon = station_coordinates[i - 1]  # Index adjustment for station_coordinates
                        standardized_data.extend(standardize_aer([aer_data], lat, lon)[0])

                # Write standardized data to output file
                output_row = {field: value for field, value in zip(fieldnames_all, row.values())}
                for field, value in zip(fieldnames_standardized, standardized_data):
                    output_row[field] = value
                writer.writerow(output_row)

# Input and output file paths
input_file = 'aer_data/combined_aer_data.csv'
output_file = 'aer_data/standardized_aer_data.csv'

# Process AER data and write to output file
process_aer_data(input_file, output_file)
