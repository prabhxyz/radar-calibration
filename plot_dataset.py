import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create the directory if it doesn't exist
save_dir = "graphics/dataset_visual"
os.makedirs(save_dir, exist_ok=True)

# Load the dataset
data = pd.read_csv('aer_data/standardized_aer_data.csv')

# Iterate over the first 500 rows
for idx in range(500):
    # Take a single row from the dataset
    selected_row = data.iloc[[idx]]

    # Extract X and y values from the selected row
    X = selected_row[['azimuth_1', 'elevation_1', 'range_1',
                      'azimuth_2', 'elevation_2', 'range_2',
                      'azimuth_3', 'elevation_3', 'range_3']]
    y = selected_row[['azimuth_4', 'elevation_4', 'range_4']]

    # Create a new figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each data point with vibrant colors and markers
    ax.scatter(X['azimuth_1'], X['elevation_1'], X['range_1'], c='red', marker='o', s=50, label='1st Radar Site')
    ax.scatter(X['azimuth_2'], X['elevation_2'], X['range_2'], c='green', marker='^', s=50, label='2nd Radar Site')
    ax.scatter(X['azimuth_3'], X['elevation_3'], X['range_3'], c='blue', marker='s', s=50, label='3rd Radar Site')
    ax.scatter(y['azimuth_4'], y['elevation_4'], y['range_4'], c='black', marker='d', s=50, label='Actual Value')

    # Connect each radar site to the actual value with lines
    for i in range(len(X)):
        ax.plot([X.iloc[i]['azimuth_1'], y.iloc[i]['azimuth_4']],
                [X.iloc[i]['elevation_1'], y.iloc[i]['elevation_4']],
                [X.iloc[i]['range_1'], y.iloc[i]['range_4']], color='red', linestyle='--')

        ax.plot([X.iloc[i]['azimuth_2'], y.iloc[i]['azimuth_4']],
                [X.iloc[i]['elevation_2'], y.iloc[i]['elevation_4']],
                [X.iloc[i]['range_2'], y.iloc[i]['range_4']], color='green', linestyle='--')

        ax.plot([X.iloc[i]['azimuth_3'], y.iloc[i]['azimuth_4']],
                [X.iloc[i]['elevation_3'], y.iloc[i]['elevation_4']],
                [X.iloc[i]['range_3'], y.iloc[i]['range_4']], color='blue', linestyle='--')

    # Customize axis labels
    ax.set_xlabel('Azimuth', fontsize=12)
    ax.set_ylabel('Elevation', fontsize=12)
    ax.set_zlabel('Range', fontsize=12)

    # Set grid lines
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # Set viewing angle
    ax.view_init(elev=30, azim=135)

    # Add a legend
    ax.legend()

    # Save the plot as an image
    plt.savefig(os.path.join(save_dir, f'plot_{idx}.png'))
    print(f"Plot {idx} saved successfully!")

    # Close the plot to prevent it from being displayed
    plt.close()
