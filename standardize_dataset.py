import csv
import os

# Specify the directory where the input CSV files are located
input_dir = 'aer_data/rounded_times'

# Specify the output file name and path
output_file = 'aer_data/combined_aer_data.csv'

# Initialize lists to hold the data from each file
file1_data = []
file2_data = []
file3_data = []
file4_data = []

# Read the data from each input file
for i, filename in enumerate(['rounded_CR_AER.csv', 'rounded_TP_AER.csv', 'rounded_WS_AER.csv', 'rounded_MS_AER.csv'], start=1):
    filepath = os.path.join(input_dir, filename)
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        data = [row for row in reader]
        if i == 1:
            file1_data = data
        elif i == 2:
            file2_data = data
        elif i == 3:
            file3_data = data
        else:
            file4_data = data

# Find the maximum number of rows across all input files
max_rows = max(len(file1_data), len(file2_data), len(file3_data), len(file4_data))

# Write the combined data to the output file
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write the header row
    header = ['Time (UTCG)', 'azimuth_1', 'elevation_1', 'range_1', 'azimuth_2', 'elevation_2', 'range_2', 'azimuth_3', 'elevation_3', 'range_3', 'azimuth_4', 'elevation_4', 'range_4']
    writer.writerow(header)

    # Write the data rows
    for i in range(max_rows):
        row = []
        if i < len(file1_data):
            row.extend([file1_data[i][0]])  # Time (UTCG) from file1
            row.extend(file1_data[i][1:])  # Azimuth, Elevation, Range from file1
        else:
            row.extend([''] * 4)  # Fill with empty strings if row doesn't exist

        if i < len(file2_data):
            row.extend(file2_data[i][1:])  # Azimuth, Elevation, Range from file2
        else:
            row.extend([''] * 3)  # Fill with empty strings if row doesn't exist

        if i < len(file3_data):
            row.extend(file3_data[i][1:])  # Azimuth, Elevation, Range from file3
        else:
            row.extend([''] * 3)  # Fill with empty strings if row doesn't exist

        if i < len(file4_data):
            row.extend(file4_data[i][1:])  # Azimuth, Elevation, Range from file4
        else:
            row.extend([''] * 3)  # Fill with empty strings if row doesn't exist

        # Write the row only if it doesn't contain any empty columns
        if '' not in row:
            writer.writerow(row)

print(f"Combined data written to {output_file}")
