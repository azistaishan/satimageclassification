import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report, accuracy_score
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import rasterio as rio
# Define the input layer

class NeuralNet:
    def __init__(self, dfFpath=None, extract=False, modelFpath=None):
        self.encodingShift = 1
        self.extractState = False
        # Load model option
        #TODO
        if dfFpath is not None:
            self.df = pd.read_csv(dfFpath)
        if extract is True:
            self.extractxy()
            print('Training and testing datasets extracted')
        if modelFpath is not None:
            self.loadModel(modelFpath)
        else:
            self.Model1()
    def extractxy(self, oversample=False, undersample=False, scale=False, testSize=0.2, randomState=42):
        """
        #TODO
        """
        df = self.df
        self.x = df.iloc[:,1:].values
        self.y = df.iloc[:,0].values
        classes = np.unique(self.y)
        self.numClasses = len(classes)
        if oversample is True:
            ros = RandomOverSampler(random_state=randomState)
            self.x, self.y = ros.fit_resample(self.x,self.y)
        if undersample is True:
            rus = RandomUnderSampler(random_state=randomState)
            self.x,self.y = rus.fit_resample(self.x, self.y)
        self.label_Encoder = LabelEncoder()
        self.y = self.label_Encoder.fit_transform(self.y)
        # Set the input size
        cols = self.df.columns
        # Assume "Class" is in columns so num of params = 1 less than col nums
        self.numParams = len(cols)-1 # Adjust this based on your input data size
        # input_layer = tf.keras.layers.Input(shape=(input_size,))
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2,random_state=42)
        y_train = tf.one_hot(y_train, depth=self.numClasses)
        y_test = tf.one_hot(y_test, depth=self.numClasses)
        #Declare the datasets to self
        self.y_test = y_test
        self.y_train = y_train
        self.x_test = x_test
        self.x_train = x_train
        self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1000)
        self.test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1000)
        # start_time = time.time()
        # history = model.fit(train_dataset, epochs=100, validation_data=test_dataset)
        # print('Time taken to train: ', time.time()-start_time)
    def Model1(self,summary=False):
        """
        #TODO
        """
        numParams = self.numParams
        #TODO
        numClasses = self.numClasses
        #TODO
        # Define the input layer
        input_size = self.numParams  # Adjust this based on your input data size
        input_layer = tf.keras.layers.Input(shape=(input_size,))

        # Define the first hidden layer with ReLU activation
        hidden_layer_1 = tf.keras.layers.Dense(128, activation='relu')(input_layer)

        # Define the second hidden layer with ReLU activation
        hidden_layer_2 = tf.keras.layers.Dense(64, activation='relu')(hidden_layer_1)

        num_classes = self.numClasses
        # Define the output layer with softmax activation for classification
        output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(hidden_layer_2)
        # 'num_classes' should be the number of classes in your classification problem

        # Create a model
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        if summary is True:
            model.summary()
    def classification_model():
        #model.fit(X_train_scaled,y_train)
        y_pred = model.predict(x_test)
        print(self.y_test)
        self.y_test = np.argmax(self.y_test, axis=1)
        print("Shape of y_test: ", self.y_test.shape)
        y_pred = np.argmax(y_pred, axis=1)
        print("Shape of y_pred: ", y_pred.shape)
        scoree = round(accuracy_score(self.y_test,y_pred)*100,2)

        f1_s = round(f1_score(self.y_test,y_pred,average='micro', zero_division=0)*100,2)

        # cross_v = cross_val_score(model,X,y,cv=10,scoring='accuracy').mean()

        # roc_ = multiclass_roc_auc_score(y_test,y_pred)
        # roc_ = multiclass_roc_auc_score(self.y_test,self.y_pred)

        print ("Model:",str(model).split("(")[0])
        print ("Accuracy Score:",scoree)
        print ("f1 Score:",f1_s)
        # print ("CV Score:",cross_v)
        # print ("ROC_AUC Score:",roc_)

        #shows the classification report
        class_report = classification_report(self.y_test, y_pred)
        print (class_report)
        # shows the confusion matrix
        # sns.heatmap(confusion_matrix(self.y_test,self.y_pred), annot=True,square=True)
    def fit(self,):
        start_time = time.time() 
        self.history = self.model.fit(self.train_dataset, 
        epochs=100, validation_data=self.test_dataset)
        self.timeTaken = time.time()-start_time

    def saveModel(self, outFpath):
        """
        Save weights of the model
        """
        self.model.save(outFpath)

    def loadModel(self, inFpath):
        """
        Load model 
        """
        self.model = tf.keras.models.load_model(inFpath)
        self.model.summary()

    def classifyImg(self, imgFpath, outFpath, modelFpath=None, noclassThres=None):
        """
        Creates the classification image 

        Parameters:

        imgFpath: Str
            Path to the image to classify
        outFpath: Str
            Path to the output file to save
        dateFpath: Str/None
            Path to date file if the imgFpath is a time series data
        modelFpath: Str/None
            Joblib file to the model created before. If None, saved model is searched
        
        Returns:
            None
        #TODO
        """
        with rio.open(imgFpath) as dst:
            img = dst.read()
            meta = dst.meta
            profile = dst.profile
        flatArr = img.reshape(img.shape[0],-1).T
        out_1 = self.model.predict(flatArr)
        out_2 = np.argmax(out_1, axis=1)
        out_2 += self.encodingShift
        classOut = out_2.reshape(img.shape[1:])
        profile.update(count=1)
        with rio.open(outFpath, "w", **profile) as dst:
            dst.write(classOut, 1)
        # return out 
if __name__ == "__main__":
    trainingcsv = r"D:\NewImage\Simulation\TestData\trainingSample3.csv"
    rasterFile = r"D:\NewImage\Simulation\TestData\stackimg\ndvi_corrected_stack30D.tif"
    outRaster = r"D:\NewImage\Simulation\TestData\ClassOut\NNout20231129.tif"
    modelOutPath = r"D:\NewImage\Simulation\ModelOut\NeuralNet20231129\MyModel1"
    def testcase1():
        test = NeuralNet(dfFpath=trainingcsv, extract=True)
    def testcase2():
        test = NeuralNet(modelFpath=modelOutPath)
        test.classifyImg(imgFpath=rasterFile, outFpath=outRaster)
    testcase2()
