from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
import os
import numpy as np
import seaborn as sns
import rasterio as rio
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
import time
#from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
from rasterio.mask import mask
import geopandas as gpd
import glob
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from maxlikelihood import MLClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import time
class Classify:
    """
    Set of methods required to train and test the cl
    """
    def __init__(self,dfFpath = None, extract=False, modelFpath=None):
        self.encodingShift = 1
        self.extractState = False
        if modelFpath is not None:
            self.model = joblib.load(modelFpath)
        if dfFpath is not None:
            self.df = pd.read_csv(dfFpath)
        if extract is True:
            self.extractxy()
            print('Training and Testing datasets extracted')
    def extractxy(self,oversample=False, undersample=False, scale=False, testSize=0.2, randomState=42):
        """
        The expected format of the data is "Class", "Val_0",---,"Val_30","Day_0",---,"Day_30"

        Parameters:

        oversample: Bool
            Option to oversample data
        undersample: Bool
            Option to undersample data
        scale: Bool
            Option to scale the data
        testSize: float
            Fraction of test data to create. 0.2 means 20% of data will be used as test data
        randomState: Int
            Seed value for random state

        Returns:

            None
        """
        #TODO Improve this part of the code
        while self.df.keys()[0] != "Class":
            toPop = self.df.keys()[0]
            print("The first column is not class")
            self.df.pop(toPop)
        self.labelEncode()
        print(self.df.keys())
        self.x = self.df.iloc[:,1:].values
        self.y = self.df.iloc[:,0].values
        
        # Oversampling/Undersampling State
        if oversample is True:
            ros = RandomOverSampler(random_state=randomState)
            self.x, self.y = ros.fit_resample(self.x,self.y)
        if undersample is True:
            rus = RandomUnderSampler(random_state=randomState)
            self.x,self.y = rus.fit_resample(self.x, self.y)
        
        # Split data:
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=testSize, random_state=randomState)
        if scale is True:
            scaler = StandardScaler()
            self.x_train = scaler.transform(self.x_train) # Or fit_transform ?
            self.x_test = scaler.transform(self.x_test) # Or fit_transform ? 
        self.extractState = True
    def randomForest(self, params= None, n_jobs=-1):

        rfc = RandomForestClassifier(n_jobs=n_jobs, verbose=1)
        if params is None:
            params = {
            'n_estimators': [100, 200, 300],  # Number of decision trees in the forest
            'max_depth': [None, 5, 10],       # Maximum depth of the decision trees
            'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
            'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
            'criterion':['gini','entropy'],        
            }
        gsCV_rfc = GridSearchCV(rfc,params,cv=3,scoring='accuracy')
        gsCV_rfc.fit(self.x_train,self.y_train)
        para_n=gsCV_rfc.best_params_
        best_rbf_classifier=gsCV_rfc.best_estimator_
        self.rfcModel = best_rbf_classifier
        self.model = self.rfcModel

    def maximumLikelihood(self,):
        mlc = MLClassifier()
        mlc.fit(self.x_train, self.y_train)
        print(mlc.score(self.x_test, self.y_test))
        self.maxlikehood = mlc

    def saveModel(self,saveFpath):
        joblib.dump(self.rfcModel, saveFpath)
    
    def loadModel(self, loadFpath):
        self.model = joblib.load(loadFpath)
    def multiclass_roc_auc_score(self,y_test,y_pred):
        y_test_new = LabelBinarizer().fit_transform(y_test)
        y_pred_new = LabelBinarizer().fit_transform(y_pred)
        return round(roc_auc_score(y_test_new,y_pred_new)*100,2)
    def classification_model(self,model=None):
        if model is None:
            model = self.model
        #model.fit(X_train_scaled,y_train)
        self.y_pred = model.predict(self.x_test)
    
        scoree = round(accuracy_score(self.y_test,self.y_pred)*100,2)
    
        f1_s = round(f1_score(self.y_test,self.y_pred,average='micro', zero_division=0)*100,2)
    
        # cross_v = cross_val_score(model,X,y,cv=10,scoring='accuracy').mean()
    
        # roc_ = multiclass_roc_auc_score(self.y_test,y_pred)
        roc_ = self.multiclass_roc_auc_score(self.y_test,self.y_pred)
    
        print ("Model:",str(model).split("(")[0])
        print ("Accuracy Score:",scoree)
        print ("f1 Score:",f1_s)
        # print ("CV Score:",cross_v)
        print ("ROC_AUC Score:",roc_)
    
        #shows the classification report
        class_report = classification_report(self.y_test,model.predict(self.x_test))
        print (class_report)
        # shows the confusion matrix
        sns.heatmap(confusion_matrix(self.y_test,self.y_pred), annot=True,square=True)   
    def labelEncode(self, colName = "Class"):
        """
        Performs label encoding if required

        Parameters:
            colName: Str
                Name of the column to be label encoded
        
        Returns:
            None
        """
        #TODO pop "Unamed 0" from the dataframe
        self.label_Encoder = LabelEncoder()
        encodedCol = self.label_Encoder.fit_transform(self.df[colName])+self.encodingShift
        self.df[colName] = encodedCol

    def getDateToJulian(self, dateFpath=None):
        """
        Changes the date file to julian date

        Parameters:
            dateFpath: Str/None
                Path to date file
        """
        dates = np.loadtxt(dateFpath, dtype='str')
        newDates = [datetime.strptime(d,"%Y%m%d").date() for d in dates]
        firstYear = newDates[0].year
        firstDay = datetime(firstYear, 1,1).date()
        julian = [(d-firstDay).days for d in newDates]
        self.dates = newDates
        self.julianDates = julian

    def classifyimg(self, imgFpath, outFpath, dateFpath = None, modelFpath=None, noClassThres=None):
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
        """
        # if self.extractState is False:
            # self.extractxy()
        # if self.y_train.dtype == "O":
            # print("The prediction is string type")
        if modelFpath is not None:
            self.model = joblib.load(modelFpath)
        # TODO if model is already there, then not required
        with rio.open(imgFpath) as dst:
            img = dst.read()
            meta = dst.meta
            profile = dst.profile
        if dateFpath is not None:
            print('Date file given, timeseries assumed')
            self.getDateToJulian(dateFpath=dateFpath)
            try:
                len(self.dates) == img.shape[0]
            except ValueError:
                print("Length mismatch of image size and dates")
            temp = np.zeros(img.shape)
            for i in range(len(self.julianDates)):
                temp[i] = self.julianDates[i]
            inputArr = np.vstack((img,temp))
            flatArr = inputArr.reshape(inputArr.shape[0],-1).T
        else:
            print('Datefile not given, image not a time series')
            flatArr = img.reshape(img.shape[0],-1).T
        predictions = self.model.predict(flatArr)
        self.predProb = self.model.predict_proba(flatArr)
        if predictions.dtype =='O':
            print('Predictions are string type')
            predictions = self.label_Encoder.fit_transform(predictions)
        if min(set(predictions)) == 0:
            print('0 cannot be a class as this messes with the nodata value')
            # This will mess with the no data value, hence increase by encodingShift
            predictions += self.encodingShift
        if noClassThres is not None:
            classMaxProb = np.max(self.predProb, axis=1)
            thresIdxs = np.where(classMaxProb<=noClassThres)[0]
            # If the prediction yields class as 5 then +1 gives class as 6. Should be 9 here
            # If class numbers as 8
            predictions[thresIdxs] = max(predictions)+1
        predictions = predictions.reshape(img.shape[1:])
        meta.update(count=1)
        profile.update(count=1)
        temp = {'dates':[1,2,3,4,5,6,7]}
        with rio.open(outFpath,"w", **profile) as dst:
            dst.write(predictions,1)
            dst.update_tags(**temp)

if __name__ == "__main__":
    def testcase1():
        """
        Single Image RFC Classification test
        """
        trainingcsv = r"D:\NewImage\Simulation\TestData\trainingSample.csv"
        test = Classify(dfFpath = trainingcsv, extract=True)
        print('Starting training')
        start_time = time.time()
        test.randomForest()
        modelOutPath = r"D:\NewImage\Simulation\ModelOut\Run20231121\rfc_Model_singleImg.joblib"
        print(f'Time taken is {time.time()-start_time}')
        print('Saving model')
        test.saveModel(saveFpath = modelOutPath)
        #Model testing of Random Forest
        test.classification_model(test.rfcModel)
    def testcase0():
        """
        Time series NDVI Classification 
        """
        trainingcsv = r"D:\NewImage\Simulation\TestData\trainingSample3.csv"
        test = Classify(dfFpath = trainingcsv, extract=True)
        print('Starting training')
        start_time = time.time()
        test.randomForest()
        modelOutPath = r"D:\NewImage\Simulation\ModelOut\Run20231121\rfc_Model_ndvitimeseries.joblib"
        print(f'Time taken is {time.time()-start_time}')
        print('Saving model')
        test.saveModel(saveFpath = modelOutPath)
        #Model testing of Random Forest
        test.classification_model(test.rfcModel)

    def testcase2():
        modelFile = r"D:\NewImage\Simulation\TestData\Models\rfc_Model_20230212.joblib"
        rasterFile = r"D:\Images\rgbnir_new\20230212T050931_20230212T051514_T44QKF.tif"
        outFile = r"D:\NewImage\Simulation\TestData\ClassOut\temp.tif"
        csvFile = r"D:\NewImage\Simulation\TestData\trainingSample.csv"
        test = Classify(dfFpath=csvFile, extract=True, modelFpath=modelFile)
        test.classification_model()
        test.classifyimg(imgFpath=rasterFile, outFpath=outFile, noClassThres=0.4)
        return test
    def testcase3():
        """
        Test for time series image
        """
        modelFile = r"D:\Ishan\imageProcessing\TestData\Models\rfcModel.joblib"
        rasterFile = r"D:\Ishan\imageProcessing\TestData\stackimg\ndvi_corrected_stack30D.tif"
        outFile = r"D:\Ishan\imageProcessing\TestData\ClassOut\rvcTimeSeriesClassification.tif"
        csvFile = r"D:\Ishan\imageProcessing\TestData\trainingSample2.csv"
        dateFile = r"D:\Ishan\imageProcessing\TestData\dates2.txt"
        test = Classify(dfFpath=csvFile, extract=True, modelFpath=modelFile)
        #test.classification_model()
        test.classifyimg(imgFpath=rasterFile, outFpath=outFile, dateFpath=dateFile)
    def testcase4():
        """
        Test for max likelihood
        """
        csvFile = r"D:\Ishan\imageProcessing\TestData\trainingSample.csv"
        test = Classify(dfFpath=csvFile, extract=True)
        test.maximumLikelihood()
        #test.classification_model(model = test.maxlikehood)
    def testcase5():
        """
        Build the models from the simulated data csv files
        """
        trainingcsv = r"D:\NewImage\Simulation\TrainingData\sigma-dot25.csv"
        modelOutPath = r"D:\NewImage\Simulation\ModelOut\sigma-dot25.joblib"
        test = Classify(dfFpath = trainingcsv, extract=True)
        print('Starting training')
        test.randomForest()
        print('Saving model')
        test.saveModel(saveFpath = modelOutPath)
    
    def testcase6():
        """
        Check the classification accuracy for the cases:
        """
        modelInPath = r"D:\NewImage\Simulation\ModelOut\sigma-dot25.joblib"
        trainincsv = r"D:\NewImage\Simulation\TrainingData\sigma-dot25.csv"
        test = Classify(dfFpath=trainincsv, modelFpath=modelInPath, extract=True)
        print(test.model)
        test.classification_model()
    # Run Testcases:
    test = testcase0()