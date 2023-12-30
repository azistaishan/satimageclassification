# For 1D Image
# trainingcsv1D = r"/home/ishan/workingDir/satImageClassification/InputDatasets/trainingSample.csv"
trainingcsv1D = "/home/ishan/workingDir/AFR/TrainingData/trainingdata.csv"
inRaster1D = r"/home/ishan/workingDir/satImageClassification/InputDatasets/rgbnir_new/20230212T050931_20230212T051514_T44QKF.tif"
# inRaster1D = "/home/ishan/workingDir/AFR/ClippedRaster/1r121_5b_composite_clipped.tif"
# For 30D Image
trainingcsv30D = r"/home/ishan/workingDir/satImageClassification/InputDatasets/trainingSample3.csv"
inRaster30D = r"/home/ishan/workingDir/satImageClassification/InputDatasets/ndvi_corrected_stack30D.tif"

# Settings for Random Forest
rfcModelOutPath1D = r"/home/ishan/workingDir/satImageClassification/RandomForest/Model/RFCModel1D.joblib"
rfcModelInPath1D = r"/home/ishan/workingDir/satImageClassification/RandomForest/Model/RFCModel1D.joblib"
rfcOutRaster1D = r"/home/ishan/workingDir/satImageClassification/RandomForest/Image/RFCRaster1D.tif"
rfcModelOutPath30D = r"/home/ishan/workingDir/satImageClassification/RandomForest/Model/RFCModel30D.joblib"
rfcModelInPath30D = r"/home/ishan/workingDir/satImageClassification/RandomForest/Model/RFCModel30D.joblib"
rfcOutRaster30D = r"/home/ishan/workingDir/satImageClassification/RandomForest/Image/RFCRaster30D.tif"
rfcParams = {
        'n_estimators': [200],  # Number of decision trees in the forest
        'max_depth': [10],       # Maximum depth of the decision trees
        'min_samples_split': [50],  # Minimum number of samples required to split a node
        'min_samples_leaf': [50],    # Minimum number of samples required to be at a leaf node
        'criterion':['gini'],        
        }

#Settings for Support Vector Machine
svmModelOutPath1D = r"/home/ishan/workingDir/satImageClassification/SupportVectorMachine/Model/svmModel1D.joblib"
svmModelInPath1D = r"/home/ishan/workingDir/satImageClassification/SupportVectorMachine/Model/svmModel1D.joblib"
svmModelOutPath30D = r"/home/ishan/workingDir/satImageClassification/SupportVectorMachine/Model/svmModel30D.joblib"
svmModelInPath30D = r"/home/ishan/workingDir/satImageClassification/SupportVectorMachine/Model/svmModel30D.joblib"
svmOutRaster1D = r"/home/ishan/workingDir/satImageClassification/SupportVectorMachine/Image/svmRaster1D.tif"
svmOutRaster30D = r"/home/ishan/workingDir/satImageClassification/SupportVectorMachine/Image/svmRaster30D.tif"
svmParams = {
        'estimator__C': [0.1, 1, 10],
        'estimator__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'estimator__gamma': ['scale', 'auto']
            }
# Settings for MaxLikelihood
maxLModelOutPath1D = r"/home/ishan/workingDir/satImageClassification/MaxLikelihood/Model/maxLModel1D.joblib"
maxLModelInPath1D = r"/home/ishan/workingDir/satImageClassification/MaxLikelihood/Model/maxLModel1D.joblib"
maxLModelModelOutPath30D = r"/home/ishan/workingDir/satImageClassification/MaxLikelihood/Model/maxLModel30D.joblib"
maxLModelModelInPath30D = r"/home/ishan/workingDir/satImageClassification/MaxLikelihood/Model/maxLModel30D.joblib"
maxLModelOutRaster1D = r"/home/ishan/workingDir/satImageClassification/MaxLikelihood/Image/maxLRaster1D.tif"
maxLModelOutRaster30D = r"/home/ishan/workingDir/satImageClassification/MaxLikelihood/Image/maxLRaster30D.tif"

# Settings for Neural Network
nnModelOutPath1D = r"/home/ishan/workingDir/satImageClassification/NeuralNet/Model/nnModel1D"
nnModelInPath1D = r"/home/ishan/workingDir/satImageClassification/NeuralNet/Model/nnModel1D"
nnModelModelOutPath30D = r"/home/ishan/workingDir/satImageClassification/NeuralNet/Model/nnModel30D"
nnModelModelInPath30D = r"/home/ishan/workingDir/satImageClassification/NeuralNet/Model/nnModel30D"
nnOutRaster1D = r"/home/ishan/workingDir/satImageClassification/NeuralNet/Image/nnRaster1D.tif"
nnOutRaster30D = r"/home/ishan/workingDir/satImageClassification/NeuralNet/Image/nnRaster30D.tif"

# Settings for Kmeans
kmeansOutGraphs1D = r"/home/ishan/workingDir/satImageClassification/KMeans/Graphs/Graphs1D"
kmeansOutGraphs30D = r"/home/ishan/workingDir/satImageClassification/KMeans/Graphs/Graphs30D"
kMeansOutRaster1D = r"/home/ishan/workingDir/satImageClassification/KMeans/Image/kMeansRaster1D.tif"
kMeansOutRaster30D = r"/home/ishan/workingDir/satImageClassification/KMeans/Image/kMeansRaster30D.tif"