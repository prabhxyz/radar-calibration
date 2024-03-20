import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

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

# Load the trained model
model = load_model('radar_calibration_model.h5')

# Normalize the example input (if necessary)
# example_input_normalized = (example_input - X_train.mean()) / X_train.std()

# Make predictions
predictions = model.predict(example_input)

# Print the predictions
print("Predicted azimuth:", predictions[0][0])
print("Predicted elevation:", predictions[0][1])
print("Predicted range:", predictions[0][2])
