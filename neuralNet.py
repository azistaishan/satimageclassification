import tensorflow as tf

# Define the input layer
input_size = 30  # Adjust this based on your input data size
input_layer = tf.keras.layers.Input(shape=(input_size,))

# Define the first hidden layer with ReLU activation
hidden_layer_1 = tf.keras.layers.Dense(128, activation='relu')(input_layer)

# Define the second hidden layer with ReLU activation
hidden_layer_2 = tf.keras.layers.Dense(64, activation='relu')(hidden_layer_1)

num_classes = 8
# Define the output layer with softmax activation for classification
output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(hidden_layer_2)
# 'num_classes' should be the number of classes in your classification problem

# Create a model
model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model's architecture
model.summary()

# Get Data:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
trainingcsv = r"D:\NewImage\Simulation\TestData\trainingSample3.csv"
df = pd.read_csv(trainingcsv)
data_x = df.iloc[:,1:].values
data_y = df.iloc[:,0].values
label_Encoder = LabelEncoder()
data_y = label_Encoder.fit_transform(data_y)
# data_y = data_y.T
# data_y = np.reshape(data_y, (len(data_y),1))
# data_y = tf.one_hot(data_y, depth=8)
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2,random_state=42)
y_train = tf.one_hot(y_train, depth=8)
y_test = tf.one_hot(y_test, depth=8)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1000)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1000)
history = model.fit(train_dataset, epochs=100, validation_data=test_dataset)