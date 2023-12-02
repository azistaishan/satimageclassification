import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report, accuracy_score
import time
# Define the input layer
input_size = 4  # Adjust this based on your input data size
input_layer = tf.keras.layers.Input(shape=(input_size,))

# Define the first hidden layer with ReLU activation
hidden_layer_1 = tf.keras.layers.Dense(128, activation='relu')(input_layer)

# Define the second hidden layer with ReLU activation
hidden_layer_2 = tf.keras.layers.Dense(64, activation='relu')(hidden_layer_1)

num_classes = 4
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
trainingcsv = r"D:\NewImage\Simulation\TestData\trainingSample.csv"
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
start_time = time.time()
history = model.fit(train_dataset, epochs=100, validation_data=test_dataset)
print('Time taken to train: ', time.time()-start_time)
def classification_model(model=model, y_test=y_test):
    #model.fit(X_train_scaled,y_train)
    y_pred = model.predict(x_test)
    print(y_test)
    y_test = np.argmax(y_test, axis=1)
    print("Shape of y_test: ", y_test.shape)
    y_pred = np.argmax(y_pred, axis=1)
    print("Shape of y_pred: ", y_pred.shape)
    scoree = round(accuracy_score(y_test,y_pred)*100,2)
    
    f1_s = round(f1_score(y_test,y_pred,average='micro', zero_division=0)*100,2)
    
    # cross_v = cross_val_score(model,X,y,cv=10,scoring='accuracy').mean()
    
    # roc_ = multiclass_roc_auc_score(y_test,y_pred)
    # roc_ = multiclass_roc_auc_score(self.y_test,self.y_pred)
    
    print ("Model:",str(model).split("(")[0])
    print ("Accuracy Score:",scoree)
    print ("f1 Score:",f1_s)
    # print ("CV Score:",cross_v)
    # print ("ROC_AUC Score:",roc_)
    
    #shows the classification report
    class_report = classification_report(y_test, y_pred)
    print (class_report)
    # shows the confusion matrix
    # sns.heatmap(confusion_matrix(self.y_test,self.y_pred), annot=True,square=True)
classification_model()