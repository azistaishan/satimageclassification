import numpy as np
import ipdb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import rasterio as rio
import ipdb
import time
class MLClassifier:
    """
    Classify satellite image data using maximum likelihodd classifier:
    
    Attributes:
    -----------
    dfFpath: str
        Path to the csv file used to train data
    extract: bool
        Option to extract training and testing data from csv file

    Methods:
    --------
        extractxy
        labelEncode
        fit
        _class_likelihood
        predictSingle
        predict
        score
        multiclass_roc_auc_score
        classifyimg

    """
    def __init__(self,dfFpath = None, extract=False):
        """
        Constructs all the necessary attributes for the object

        Parameters
        ----------
        dfFpath: str
            Path to the csv file used to train data
        extract: bool
            Option to extract training and testing data from csv file
        """
        self.encodingShift = 1  # LabelEndoder output increased by 1 so that no 0 class encoded. Mixes with no-data val
        if dfFpath is not None:
            self.df = pd.read_csv(dfFpath)
        if extract is True:
            self.extractxy()
    def extractxy(self,oversample=False, undersample=False, scale=False, testSize=0.2, randomState=42):
        """
        Extracts data from the csv file
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
        encodedCol = self.label_Encoder.fit_transform(self.df[colName])
        encodedCol += self.encodingShift
        self.df[colName] = encodedCol

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        Fits the data 

        Parameters:
            x - numpy array of shape (n, d); n = #observations; d = #variables
            y - numpy array of shape (n,)
        
        Returns:
            None
        '''
        # no. of variables / dimension
        self.d = x.shape[1]
        
        # no. of classes; assumes labels to be integers from 0 to nclasses-1
        classes = list(set(y)) 
        classes.sort()
        self.classesInt = classes
        self.nclasses = len(self.classesInt)
        
        # list of means; mu_list[i] is mean vector for label i
        self.mu_list = []
        
        # list of inverse covariance matrices;
        # sigma_list[i] is inverse covariance matrix for label i
        # for efficiency reasons we store only the inverses
        self.sigma_inv_list = []
        
        # list of scalars in front of e^...
        self.scalars = []
        
        n = x.shape[0]
        self.sigma = []
        for i in range(self.nclasses):
            
            # subset of obesrvations for label i
            cls_x = np.array([x[j] for j in range(n) if y[j] == self.classesInt[i]])
            #ipdb.set_trace()
            mu = np.mean(cls_x, axis=0)
            
            # rowvar = False, this is to use columns as variables instead of rows
            sigma = np.cov(cls_x, rowvar=False)
            self.sigma.append(sigma)
            if np.sum(np.linalg.eigvals(sigma) <= 0) != 0:
                # if at least one eigenvalue is <= 0 show warning
                # print(f'Warning! Covariance matrix for label {cls} is not positive definite!\n')
                print(f'Warning! Covariance matrix for label {i} is not positive definite!\n')
            
            sigma_inv = np.linalg.inv(sigma)
            
            scalar = 1/np.sqrt(((2*np.pi)**self.d)*np.linalg.det(sigma))
            
            self.mu_list.append(mu)
            self.sigma_inv_list.append(sigma_inv)
            self.scalars.append(scalar)
            # ipdb.set_trace()
    def _class_likelihood(self, x: np.ndarray, cls: int) -> float:
        '''
        Intermediate method to fit the data

        Parameters:
            x - numpy array of shape (d,)
            cls - class label
        
        # Classes label encoding is increased by 1
        # Hence here, the index is taken by decreasing cls value by 1
        # Assumption, classes exist without a drop of value between them
        # Hence 0-7 -> 1-8 -> 0-7
        
        Returns: likelihood of x under the assumption that class label is cls
        '''
        # Classes label encoding is increased by 1
        # Hence here, the index is taken by decreasing cls value by 1
        # Assumption, classes exist without a drop of value between them
        # Hence 0-7 -> 1-8 -> 0-7
        #TODO change getting idx value, from class, get index of class
        idx = cls-1
        mu = self.mu_list[idx]
        sigma_inv = self.sigma_inv_list[idx]
        scalar = self.scalars[idx]
        d = self.d
        # ipdb.set_trace()
        exp = (-1/2)*np.dot(np.matmul(x-mu, sigma_inv), x-mu)
        # print(exp)
        return scalar * (np.e**exp)
    
    def predictSingle(self, x: np.ndarray) -> int:
        '''
        x - numpy array of shape (d,)
        Returns: predicted label
        '''
        # likelihoods = [self._class_likelihood(x, i) for i in range(self.nclasses)]
        likelihoods = [self._class_likelihood(x, i) for i in self.classesInt]
        #TODO Get the likelihood to prepare for the no-class issue
        outIdx = np.argmax(likelihoods)
        return self.classesInt[outIdx]

    def _predictProb(self, x: np.ndarray) -> int:
        '''
        x - numpy array of shape (d,)
        Returns: predicted label
        '''
        # likelihoods = [self._class_likelihood(x, i) for i in range(self.nclasses)]
        likelihoods = [self._class_likelihood(x, i) for i in self.classesInt]
        #TODO Get the likelihood to prepare for the no-class issue
        # outIdx = np.argmax(likelihoods)
        return likelihoods
    def predict(self, x):
        out = map(self.predictSingle, x)        
        return np.array(list(out))

    def predict_proba(self, x):
        out = map(self._predictProb, x) 
        return np.array(list(out))

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        '''
        x - numpy array of shape (n, d); n = #observations; d = #variables
        y - numpy array of shape (n,)
        Returns: accuracy of predictions
        '''
        n = x.shape[0]
        predicted_y = np.array([self.predict(x[i]) for i in range(n)])
        n_correct = np.sum(predicted_y == y)
        return n_correct/n
    
    def multiclass_roc_auc_score(self,y_test,y_pred):
        y_test_new = LabelBinarizer().fit_transform(y_test)
        y_pred_new = LabelBinarizer().fit_transform(y_pred)
        return round(roc_auc_score(y_test_new,y_pred_new)*100,2)
    
    def classification_model(self):
        """
        Get the classification accuracy results of the model

        """
        #model.fit(X_train_scaled,y_train)
        y_pred = self.predict(self.x_test)
    
        scoree = round(accuracy_score(self.y_test,y_pred)*100,2)
    
        f1_s = round(f1_score(self.y_test,y_pred,average='micro')*100,2)
    
        # cross_v = cross_val_score(model,X,y,cv=10,scoring='accuracy').mean()
    
        roc_ = self.multiclass_roc_auc_score(self.y_test,y_pred)
    
        # print ("Model:",str(model).split("(")[0])
        print ("Accuracy Score:",scoree)
        print ("f1 Score:",f1_s)
        # print ("CV Score:",cross_v)
        print ("ROC_AUC Score:",roc_)
    
        #shows the classification report
        class_report = classification_report(self.y_test,self.predict(self.x_test))
        print (class_report)
        # shows the confusion matrix
        sns.heatmap(confusion_matrix(self.y_test,y_pred), annot=True,square=True)   
    
    def classifyimg(self, imgFpath, outFpath, modelFpath=None):
        """
        #TODO The model has method predict. 
        Need to save the model such that model as an object is taken up 
        Hence classifyimg, score should be out of this class.
        """
        with rio.open(imgFpath) as dst:
            img = dst.read()
            meta = dst.meta
        flatArr = img.reshape(img.shape[0],-1).T
        predictions = self.predict(flatArr)
        if predictions.dtype =='O':
            print('String type prediction, converting to endoded')
            print(f'Predicted unique values: {set(predictions)}')
            predictions = self.label_Encoder.fit_transform(predictions)
        else:
            print(f'Not a string type prediction, type is {predictions.dtype}')
            print(f'Predicted unique values: {set(predictions)}')
        predictions = predictions.reshape(img.shape[1:])
        meta.update(count=1)
        with rio.open(outFpath,"w", **meta) as dst:
            dst.write(predictions,1)

if __name__ == '__main__':
    def testcase1():
        #trainingcsv = r"D:\Ishan\imageProcessing\TestData\trainingSample.csv"
        trainingcsv = r"D:\NewImage\Simulation\TrainingData\sigma-dot06.csv"
        test = MLClassifier(dfFpath=trainingcsv, extract=True)
        test.fit(x = test.x_train, y=test.y_train)
        test.classification_model()
        return test
    def testcase2():
        trainingcsv = r"D:\NewImage\Simulation\TestData\trainingSample.csv"
        rasterFile = r"D:\Images\rgbnir_new\20230212T050931_20230212T051514_T44QKF.tif"
        outFile = r"D:\Ishan\imageProcessing\TestData\ClassOut\20230212_MaxLikeHood.tif"
        test = MLClassifier(dfFpath=trainingcsv, extract=True)
        test.fit(x = test.x_train, y=test.y_train)
        #test.classifyimg(imgFpath=rasterFile, outFpath=outFile )
        return test
    def testcase3():
        """
        Accuracy test for the single band image classification
        """
        trainingcsv = r"D:\NewImage\Simulation\TestData\trainingSample.csv"
        
        test = MLClassifier(dfFpath=trainingcsv, extract=True)
        start_time = time.time()
        test.fit(x = test.x_train, y=test.y_train)
        print("Time taken for the training: ", time.time()-start_time)
        test.classification_model()
        return test

    def testcase4():
        """
        Accuracy test for the single band image classification
        """
        trainingcsv = r"D:\NewImage\Simulation\TestData\trainingSample3.csv"
        print("Running")
        test = MLClassifier(dfFpath=trainingcsv, extract=True)
        start_time = time.time()
        test.fit(x = test.x_train, y=test.y_train)
        print("Time taken for the training: ", time.time()-start_time)
        test.classification_model()
        return test
    # test = testcase2()

    test = testcase3()
