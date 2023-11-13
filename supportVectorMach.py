import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from randomforest import RFClassifiy
import joblib
class SVMClassification(Classifiy):
    """
    Classify satellite image data using Support Vector Machine

    Attributes:

    dfFpath: str
        Path to the csv file used to train data
    extract: bool
        Option to extract the training and testing data from csv file
    modellFpath: str
        Path to the saved model. Loaded model can be used to 

    """
    def __init__(self,dfFpath=None, extract=True, modelFpath=None):
        """
        Constructs all the necessary attributes for the object

        Parameters:
            dfFpath: str
                Path to the csv file used to train data
            extract: bool
                Option to extract the training and testing data from csv file
            modellFpath: str
                Path to the saved model. Loaded model can be used to 
        Returns:
            None 
        """
        if modelFpath is not None:
            self.svModel = joblib.load(modelFpath)
        RFClassifiy.__init__(self,dfFpath=dfFpath, extract=extract)
    def supportVectorMachine(self, param_grid=None):
        """
        Applies the support vector machine to the training data
        
        Parameters:
        param_grid: dictionary
            Dictionary of parameters for grid search of parameters
        
        Returns:
            None
        """
        svm_classifier = OneVsRestClassifier(SVC(probability=True))
        if param_grid is None:
            param_grid = {
                'estimator__C': [0.1, 1, 10],
                'estimator__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'estimator__gamma': ['scale', 'auto']
            }
        grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5)
        grid_search.fit(self.x_train, self.y_train)
        best_svm_classifier = grid_search.best_estimator_
        self.svModel = best_svm_classifier
        self.svmDecisionScores = best_svm_classifier.decision_function(self.x_test)

        
