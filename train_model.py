import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

# Load the dataset
data = pd.read_csv('aer_data/combined_aer_data.csv')

# Split data into features (X) and target variable (y)
X = data[['azimuth_1', 'elevation_1', 'range_1',
          'azimuth_2', 'elevation_2', 'range_2',
          'azimuth_3', 'elevation_3', 'range_3']]
y = data[['azimuth_4', 'elevation_4', 'range_4']]

# Perform feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the input data for CNN
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Split the scaled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    Conv1D(32, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01)),
    Dropout(0.2),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01)),
    Dropout(0.2),
    Dense(3)
])

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.95

lr_schedule = LearningRateScheduler(lr_scheduler)

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the neural network
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, lr_schedule])

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print('Mean Squared Error:', mse)

# Save the trained model for future use
model.save('models/radar_calibration_model.h5')