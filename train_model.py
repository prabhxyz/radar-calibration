import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# Load the dataset
data = pd.read_csv('aer_data/combined_aer_data.csv')

# Split data into features (X) and target variable (y)
X = data[['azimuth_1', 'elevation_1', 'range_1',
          'azimuth_2', 'elevation_2', 'range_2',
          'azimuth_3', 'elevation_3', 'range_3']]
y = data[['azimuth_4', 'elevation_4', 'range_4']]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert pandas DataFrames to numpy arrays
X_train_scaled = X_train_scaled.astype('float32')
X_test_scaled = X_test_scaled.astype('float32')
y_train = y_train.astype('float32').values
y_test = y_test.astype('float32').values

# Define TabNet model
tabnet_model = TabNetRegressor()

# Train the model
tabnet_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], patience=10, max_epochs=100)

# Save the trained model
torch.save(tabnet_model, 'models/radar_calibration_model.pt')