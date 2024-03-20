import csv
import os
from datetime import datetime, timedelta

# Specify the directory where the input CSV files are located
input_dir = 'aer_data'

# Specify the output directory for the new CSV files
output_dir = 'aer_data/rounded_times'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Round the time to the nearest minute
def round_time(time_str):
    time_obj = datetime.strptime(time_str, '%d %b %Y %H:%M:%S.%f')
    rounded_time = time_obj.replace(second=0, microsecond=0) + timedelta(seconds=30)
    rounded_time -= timedelta(minutes=rounded_time.minute % 1, seconds=rounded_time.second)
    return rounded_time.strftime('%d %b %Y %H:%M:%S.%f')[:-3]

# Process each input file
for filename in ['CR_AER.csv', 'TP_AER.csv', 'WS_AER.csv']:
    input_file = os.path.join(input_dir, filename)
    output_file = os.path.join(output_dir, f'rounded_{filename}')

    # Read the input file and round the times
    rounded_data = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Save the header row
        for row in reader:
            rounded_time = round_time(row[0])
            rounded_row = [rounded_time] + row[1:]
            rounded_data.append(rounded_row)

    # Remove duplicate rows after rounding
    rounded_data = list(set(map(tuple, rounded_data)))
    rounded_data = [list(row) for row in rounded_data]

    # Write the rounded data to the new output file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rounded_data)

    print(f"Rounded data written to {output_file}")