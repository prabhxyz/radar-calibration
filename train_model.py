import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data = pd.read_csv('aer_data/combined_aer_data.csv')

# Split data into features (X) and target variable (y)
X = data[['azimuth_1', 'elevation_1', 'range_1',
          'azimuth_2', 'elevation_2', 'range_2',
          'azimuth_3', 'elevation_3', 'range_3']]
y = data[['azimuth_4', 'elevation_4', 'range_4']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(3)  # Output layer with 3 neurons for azimuth, elevation, and range
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the neural network
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print('Mean Squared Error:', mse)

# Save the trained model for future use
model.save('models/radar_calibration_model.h5')