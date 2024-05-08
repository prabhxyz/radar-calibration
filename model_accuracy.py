import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Function to train the model and return metrics
def train_model(X_train_scaled, y_train, X_test_scaled, y_test, max_epochs):
    tabnet_model = TabNetRegressor()
    mse_list, rmse_list, mae_list, r2_list = [], [], [], []

    for epoch in range(max_epochs):
        print(f"Training epoch {epoch}...")
        tabnet_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], patience=10, max_epochs=epoch + 1)

        # Predictions
        y_pred = tabnet_model.predict(X_test_scaled)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Append metrics to lists
        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

    return mse_list, rmse_list, mae_list, r2_list

# Load the dataset
data = pd.read_csv('aer_data/standardized_aer_data.csv')

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

# Number of epochs
max_epochs = 50

# Train the model and get metrics
mse_list, rmse_list, mae_list, r2_list = train_model(X_train_scaled, y_train, X_test_scaled, y_test, max_epochs)

# Plotting
epochs = list(range(max_epochs))

plt.figure(figsize=(10, 8))

# MSE plot
plt.subplot(2, 2, 1)
plt.plot(epochs, mse_list, linestyle='-')
# plt.scatter(42, mse_list[41], color='red')  # Highlight the 42 epoch data point
plt.text(42, mse_list[41], f'{mse_list[41]:.2f}', verticalalignment='bottom', horizontalalignment='left', color='red')  # Add value of 42 epoch
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Mean Squared Error')

# RMSE plot
plt.subplot(2, 2, 2)
plt.plot(epochs, rmse_list, linestyle='-')
# plt.scatter(42, rmse_list[41], color='red')  # Highlight the 42 epoch data point
plt.text(42, rmse_list[41], f'{rmse_list[41]:.2f}', verticalalignment='bottom', horizontalalignment='left', color='red')  # Add value of 42 epoch
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error')

# MAE plot
plt.subplot(2, 2, 3)
plt.plot(epochs, mae_list, linestyle='-')
# plt.scatter(42, mae_list[41], color='red')  # Highlight the 42 epoch data point
plt.text(42, mae_list[41], f'{mae_list[41]:.2f}', verticalalignment='bottom', horizontalalignment='left', color='red')  # Add value of 42 epoch
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Mean Absolute Error')

# R2 plot
plt.subplot(2, 2, 4)
plt.plot(epochs, r2_list, linestyle='-')
# plt.scatter(42, r2_list[41], color='red')  # Highlight the 42 epoch data point
plt.text(42, r2_list[41], f'{r2_list[41]:.2f}', verticalalignment='bottom', horizontalalignment='left', color='red')  # Add value of 42 epoch
plt.xlabel('Epoch')
plt.ylabel('R^2')
plt.title('R-squared')

plt.tight_layout()
plt.show()