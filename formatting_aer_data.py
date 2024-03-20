import csv

def remove_chunks(filename):
    start_marker = 'Interval Statistics'
    header_marker = ["Time (UTCG)", "Azimuth (deg)", "Elevation (deg)", "Range (km)"]
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        data = [row for row in reader if row]  # Skip empty rows

    # Initialize flag
    delete_flag = False

    # Iterate through data
    new_data = []
    for row in data:
        if row[0] == start_marker:
            delete_flag = True
        elif row == header_marker:
            delete_flag = False
        elif not delete_flag:
            new_data.append(row)

    # Write back to a new CSV file
    new_filename = filename
    with open(new_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_data)

    return new_filename

# Format the .csv files
#remove_chunks('aer_data/CR_AER.csv')
#remove_chunks('aer_data/TP_AER.csv')
#remove_chunks('aer_data/WS_AER.csv')
remove_chunks('aer_data/MS_AER.csv')