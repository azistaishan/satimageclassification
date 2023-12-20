# imageProcessing
To process images for further analysis. Make composite images useful for works such as time series analysis, supervised and unsupervised classification of land

---
1. **File: plotGraps.py**
    - **getGraphs Class:**
        - Methods:
            - `Initialization (__init__):`: Initializes the class object with data either from the provided object or by loading files. Parameters:
              - `kmobj`: Object initialized using kmeansCluster.py
              - `loadFiles`: Boolean indicating whether to load files or use the provided object
              - `dateFpath`: Path to date file (used if `kmobj` is not provided)
              - `imgFpath`: Path to image file (used if `kmobj` is not provided)
              - `clusterImgFpath`: Path to cluster image file (used if `kmobj` is not provided)
            - `loadFiles`: Loads necessary files (date file, image stack, cluster image) if not provided directly during initialization.
            - `plotClusterId`: Plots trends for a specific cluster ID, allowing customization of parameters like random pixel selection, y-axis range, and plot saving options.
            - `plotAllClusters`: Plots trends for all clusters defined in the cluster file, saving each plot to the specified folder.

    The `getGraphs` class facilitates plotting graphs for k-means clustering. It allows the plotting of trends for specific cluster IDs or all clusters, offering options for random pixel selection, specifying y-axis range, and saving the generated plots to a designated folder.

---

2. **File: kmeansCluster.py**
   - **kMeansCluster Class:**
       - Methods:
           - `Initialization (__init__):`: Initializes the class object and loads image `imageFpath` and date data `dateFpath`.
           - `getImage`: Loads the image file and reshapes it for analysis.
           - `setDates`: Parses dates from the date file and calculates Julian dates.
           - `getElbowCurve`: Plots an elbow curve to determine the optimal number of clusters.
           - `getKmeansTSL`: Performs k-means clustering using the tslearn library, generates labels, and optionally saves the result as an image.
           - `getKmeansCV`: Performs k-means clustering using OpenCV, generates labels, and optionally saves the result as an image.
           - `createGraph`: Creates graphs for all clusters using the `getGraphs` class from `plotGraps.py`.

   The `kMeansCluster` class serves to perform k-means clustering on image data using either tslearn or OpenCV. It offers functionalities like generating an elbow curve to find the optimal cluster count, performing clustering, and saving the clustered images. The class also facilitates the creation of graphs for clustered data points. The `testcase0` function demonstrates the usage of this class by performing clustering on a provided image and date file.

---

3. **File: supportVectorMach.py**
   - **SVMClassification Class:**
       - Methods:
           - `Initialization (__init__):`: Initializes the class object and loads the SVM model from a saved file using joblib. Inherits attributes and methods from the `RFClassifiy` class. Parameters: 
             - `dfFpath`: Path to the CSV file used for training data.
             - `extract`: Boolean indicating the option to extract training and testing data from the CSV file.
             - `modelFpath`: Path to the saved model that can be loaded for use.
           - `supportVectorMachine`: Applies Support Vector Machine (SVM) to the training data. Utilizes `GridSearchCV` to perform hyperparameter tuning through grid search. Parameters:
               - `param_grid`: Dictionary containing parameters for grid search of SVM hyperparameters.

   The `SVMClassification` class serves to perform SVM classification on satellite image data. It includes a method `supportVectorMachine` that applies SVM to the training data, performs hyperparameter tuning using grid search, and saves the best SVM model obtained from the grid search. The class appears to inherit from `RFClassifiy` and utilizes `GridSearchCV` for hyperparameter optimization.

---

4. **File: randomForest.py**
    
    - **RandomForest Class:** This class contains methods and functionalities for training, testing, and applying Random Forest and Maximum Likelihood classification algorithms. The class initializes with options to load a pre-existing model and/or a dataset from a CSV file. 
    Methods:
            
      - `Initialization (__init__):` Initializes the object with attributes such as dfFpath (path to the CSV file used for training data), extract (option to extract training and testing data from the CSV file), and modelFpath (path to a saved model).
      - `extractxy`: Extracts features and labels from the dataset, allows oversampling/undersampling, and splits the data into training and testing sets.
      - `fit`: Trains the Random Forest classifier using hyperparameter tuning via GridSearchCV.
      - `maximumLikelihood`: Fits a Maximum Likelihood classifier to the data.
      - `saveModel`/`loadModel`: Save/load trained models using joblib.
      - `multiclass_roc_auc_score`: Calculates the ROC-AUC score for multiclass classification.
      - `classification_model`: Evaluates classification models, showing accuracy, F1 score, ROC-AUC score, classification report, and confusion matrix.
      - `labelEncode`: Performs label encoding on specified columns.
      - `getDateToJulian`: Converts date data into Julian date format.
      - `classifyimg`: Classifies raster images using the trained model and saves the output.

    - **Test Cases (In the if _ _ name_ _  == "_ _ main_ _ " block):**
        Several test cases (testcase0 to testcase6) demonstrate the usage of the RandomForest class for various scenarios such as:
        - Training and testing Random Forest models with different datasets.
        - Testing classification on single images and time series NDVI data.
        - Classification accuracy assessment and evaluation.
        - Training models and saving them for future use.

    These test cases showcase the functionalities of the RandomForest class, including model training, testing, evaluation, and application on different types of data for classification purposes.

---

5. **File: maxlikelihood.py**
   - **MLClassifier Class:**
     - Methods:
       - `Initialization (__init__):` Initializes the object with attributes such as `dfFpath` (path to the CSV file used for training data), `extract` (option to extract training and testing data from the CSV file).
       - `extractxy`: Extracts data from a CSV file, preprocesses it, and splits it into training and testing sets with options for oversampling, undersampling, and scaling.
       - `labelEncode`: Performs label encoding on specified columns.
       - `fit`: Fits the data to the Maximum Likelihood Classifier, calculating means, covariance matrices, and other necessary parameters.
       - `_class_likelihood`: Calculates the likelihood of a given data point belonging to a class.
       - `predictSingle`: Predicts a single label for a given data point.
       - `predict`: Generates predictions for given data.
       - `predict_proba`: Computes probabilities of predictions.
       - `score`: Calculates the accuracy of predictions.
       - `multiclass_roc_auc_score`: Computes the ROC-AUC score for multiclass classification.
       - `classification_model`: Evaluates the classification accuracy of the model using various metrics and visualizations.
       - `classifyimg`: Classifies raster images using the trained model and saves the classified image.

   - **Test Cases (In the if _ _ name_ _  == "_ _ main_ _ " block):**
     - Defined test cases (`testcase1` to `testcase4`) for training models, evaluating classification accuracy, and classifying raster images. Each test case demonstrates different scenarios such as training with varying datasets, image classification, and accuracy assessment.

    The file in brief contains the MLClassifier class, enabling data preprocessing, Maximum Likelihood classification, model evaluation, and image classification functionalities. Additionally, it includes test cases to showcase the class functionalities across diverse scenarios.

---

6. **File: neuralNet.py**
   - **NeuralNet Class:**
       - Methods:
         - `Initialization (__init__):` Initializes the NeuralNet class object with options to load a dataset from a CSV file (`dfFpath`), extract data for training and testing (`extract`), and load a pre-existing model (`modelFpath`).
         - `extractxy`: Extracts data from a CSV file, preprocesses it, and prepares it for training and testing, allowing options for oversampling, undersampling, and scaling.
         - `Model1`: Defines and compiles a neural network model using TensorFlow with customizable layers and activation functions.
         - `classification_model`: Evaluates the neural network model's performance using metrics like accuracy, F1 score, and classification report on test data.
         - `fit`: Trains the neural network model using training datasets.
         - `saveModel/loadModel`: Saves or loads the trained model using TensorFlow's functionalities.
         - `classifyImg`: Classifies raster images using the trained neural network model and saves the output.

   - **Test Cases (In the if _ _ name_ _  == "_ _ main_ _ " block):**
       - `testcase1`: Loads data from a CSV file, trains the neural network model, and evaluates its performance on test data.
       - `testcase2`: Loads a pre-existing model and uses it to classify raster images.

    The file essentially defines a NeuralNet class which is responsible for creating, training, evaluating, and using a neural network model for classification tasks. It also provides test cases demonstrating the training, evaluation, and image classification functionalities of the NeuralNet class.

---

7. **File: imageStack.py**
   - **ImageStack Class:**
       - Methods:
           - `Initialization (__init__):` Initializes the `ImageStack` class with methods to perform various operations on image stacks and cloud masking.
           - `image_stack`: Stacks RGB and Near-Infrared (NIR) images into a single file.
           - `rgbnirFolderstack`: Stacks RGB and NIR images from folders into individual files.
           - `FCCFolderstack`: Creates False Color Composite (FCC) images by rearranging band orders.
           - `cloudMaskTest`: Generates cloud masks based on thresholding green bands in RGB images.
           - `getcloudMask`: Creates cloud masks from RGB+NIR images based on a specified threshold value.
           - `applyCloudMask`: Applies cloud masks to images and generates fixed images by filling masked areas.
           - `getImageStack`: Stacks single-value maps (e.g., NDVI, NDRE) into multi-band image stacks.
           - `maskImageStack`: Masks specific values in an image stack using another mask image.

    The file essentially defines an `ImageStack` class responsible for image manipulation, stacking, and cloud masking. It provides methods to handle RGB, NIR, and multi-band image stacks along with cloud detection and masking functionalities.

---

8. **File: datafromvectors.py**
   - **dataFromVectors Class:**
       - Methods:
         - `Initialization (__init__):` Initializes the `dataFromVectors` class with attributes such as `shpFilePath` (path to Shape File), `rasterFilePath` (path to Raster File), `valScale` (scaling factor for values in the raster), and `dateFpath` (path to the date file).
           - `openVector`: Opens and accesses information from the specified shapefile.
           - `getKeyDict`: Creates a dictionary with keys from the shapefile.
           - `selectKeys`: Selects specific keys/properties from the shapefile.
           - `openRaster`: Opens and reads the raster file.
           - `extractData1`: Extracts data from raster and vector files, handling time series data if available.
           - `extractData`: Alternative method to extract data, handles time series and non-time series data.
           - `extractData2`: Another alternative method for extracting data.
           - `getDateToJulian`: Converts dates in the dataset to Julian dates.
           - `chooseClass`: Method stub, possibly intended for further development.
           - `saveDataFrame`: Saves the extracted data to a CSV file.

   The file primarily defines a `dataFromVectors` class, enabling the extraction and processing of data from raster and vector files. It handles both time series and non-time series data, extracting specific attributes while handling potential out-of-bounds scenarios when processing shapefiles and raster images. The code includes functionality to convert dates to Julian dates and save the extracted data as a CSV file.

---



