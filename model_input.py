import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import StandardScaler
import torch

# Example input data
example_input = pd.DataFrame({
    'azimuth_1': [0.5],
    'elevation_1': [0.3],
    'range_1': [100],
    'azimuth_2': [0.4],
    'elevation_2': [0.2],
    'range_2': [150],
    'azimuth_3': [0.6],
    'elevation_3': [0.1],
    'range_3': [200]
})

# Standardize the example input (if necessary)
scaler = StandardScaler()
example_input_scaled = scaler.fit_transform(example_input)

# Load the trained model
tabnet_model = torch.load('models/radar_calibration_model.pt')

# Make predictions
predictions = tabnet_model.predict(example_input_scaled)

# Print the predictions
print("Predicted azimuth:", predictions[0][0])
print("Predicted elevation:", predictions[0][1])
print("Predicted range:", predictions[0][2])