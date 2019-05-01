from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, Imputer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.decomposition import KernelPCA

import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

import pandas as pd
import numpy as np

import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import firestore


class GMS:
    
    maxScore = 0
    bestModel = ""
    algorithm = ""
    
    maxScoreWithHighVariance = 0
    bestModelWithHighVariance = ""
    algorithmWithHighVariance = ""
    
    
    def __init__(self, data):
        self.name = ''
        self.data = data
        self.uid = data["uid"]
        self.modelName = data["modelname"]
        self.dataset = data["dataset"]
        self.columns = data["columns"]
        self.target = data["target"]
        self.categoricalcolumns = data["categoricalcolumns"]
        self.numericalcolumns = data["numericalcolumns"]
        
        self.db = firestore.client()
        
        self.ss = StandardScaler()
        self.kpca = KernelPCA(n_components = 2, kernel = 'rbf')
    
    
    def SaveTrainStatus(self, status, detail):
        oldPost = self.db.collection(u'trainstatus').document(u'' + self.data["uid"] + "." + self.modelName)
        if status == 1:
            oldPost.delete()
        else:
            oldPost.set({"status": status, "detail": detail})
    
    
    def SaveModel(self):
        '''Save best model to memory'''
        self.SaveModelToMemory()
        
        '''Save the bestModel path to database'''
        self.SaveModelToDB()
        
    
    def SaveModelToDB(self):
        
        catCols = []
        for catColIndex in self.categoricalcolumns:
            catCols.append(dict({ "name": self.columns[catColIndex], "values": sorted(pd.unique(self.data_frame.iloc[:, catColIndex].values).tolist()) }))
        
        numCols = []
        for numColIndex in self.numericalcolumns:
            numCols.append(self.columns[numColIndex])
            
        targetCol = dict({ "name": self.columns[self.target], "values": sorted(pd.unique(self.data_frame.iloc[:, self.target].values).tolist()) })
        
        newModel = {"modelname": self.modelName, "catCols": catCols , "numCols": numCols, "targetCol": targetCol, "algorithm": self.algorithm, "accuracy": self.maxScore, "uid": self.uid}
        
        self.db.collection(u'models').add(newModel)
        
    

    def SaveModelToMemory(self):
        modelPath = self.uid + self.modelName + ".txt"
        scalerPath = self.uid + self.modelName + "scaler.txt"
        
        default_bucket = storage.bucket(name="churn-2537f.appspot.com", app=None)
        
        
        modelInBytes = pickle.dumps(self.bestModel)
        scalerInBytes = pickle.dumps(self.ss)
        
        modelBlob = default_bucket.blob(modelPath)
        modelBlob.upload_from_string(modelInBytes)
        
        scalerBlob = default_bucket.blob(scalerPath)
        scalerBlob.upload_from_string(scalerInBytes)
        
        
        
    
    def CheckMetrics(self, algorithm, classifier):
        
        y_train_predict = classifier.predict(self.X_train)
        y_train_predict = (y_train_predict > 0.5)
        
        y_test_predict = classifier.predict(self.X_test)
        y_test_predict = (y_test_predict > 0.5)
        
        accuracy_train = accuracy_score(self.y_train, y_train_predict)
        accuracy_test = accuracy_score(self.y_test, y_test_predict)
        
        cm = confusion_matrix(self.y_test, y_test_predict)
        print(cm)
        
        
        print(algorithm + " Train Accuracy:")
        print(accuracy_train)
        print(algorithm + " Test Accuracy:")
        print(accuracy_test)
        
        acc_variance = abs(accuracy_test - accuracy_train)
        acc_avg = (accuracy_train + accuracy_test) / 2
        
        
        if(acc_variance <= 6):
            if(acc_avg > self.maxScore):
                self.bestModel = classifier
                self.maxScore = acc_avg
                self.algorithm = algorithm
        else:
            if(acc_avg > self.maxScoreWithHighVariance):
                self.bestModelWithHighVariance = classifier
                self.maxScoreWithHighVariance = acc_avg
                self.algorithmWithHighVariance = algorithm
    

    def Preprocess(self):        
        ''' Make dataset a dataframe '''
        self.data_frame = pd.DataFrame(self.dataset, columns = self.columns)
        self.data_frame = self.data_frame[self.columns].apply(pd.to_numeric, errors="ignore")

        ''' Handle Missing Values in categorical columns '''
        for col in self.categoricalcolumns:
            #print(self.data_frame.iloc[:, col].value_counts())
            self.data_frame.iloc[:, col].fillna(self.data_frame.iloc[:, col].value_counts().index[0], inplace = True)
        
        ''' Assign columns'''
        self.X = self.data_frame.iloc[:, (self.categoricalcolumns + self.numericalcolumns)].values
        self.y = self.data_frame.iloc[:, self.target].values
        
        numericalRange = list(range(len(self.categoricalcolumns), len(self.categoricalcolumns) + len(self.numericalcolumns)))
        
        ''' Handle Missing Values in numerical columns '''
        imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
        imputer.fit(self.X[:, numericalRange])
        self.X[:, numericalRange] = imputer.transform(self.X[:, numericalRange])
        
        ''' Encode categorical vars '''
        self.EncodeCategoricalVars()
        
        ''' Encode y column '''
        labelEncoder = LabelEncoder()
        self.y = labelEncoder.fit_transform(self.y)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = .25, random_state = 0)
        
        ''' Scale vars '''
        self.X_train = self.ss.fit_transform(self.X_train)
        self.X_test = self.ss.transform(self.X_test)
        
        """
        ''' Applying Kernel PCA - Dimensionality Reduction '''
        self.X_train = self.kpca.fit_transform(self.X_train)
        self.X_test = self.kpca.fit(self.X_test)
        """

    
    '''Encoding categorical data'''
    def EncodeCategoricalVars(self):
        feature_list = []
        numOfUniqueValsForCatCols = []
        hasMulticlassCat = False
        indexZeroAdded = False
        
        #Check if there is categorical variable
        if len(self.categoricalcolumns) != 0:
            for i in range(0, len(self.categoricalcolumns)):
                numOfUniqueVals = len(pd.unique(self.X[:, i]).tolist())
                
                labelEncoder = LabelEncoder()
                self.X[:, i] = labelEncoder.fit_transform(self.X[:, i])
                
                ''' if there are more than 2 classes then use OHE on it '''
                if(numOfUniqueVals > 2): 
                    feature_list.append(i)
                    if(not indexZeroAdded):
                        numOfUniqueValsForCatCols.append(0)
                        indexZeroAdded = True
                    ''' save number of unique vals for every categorical column '''
                    numOfUniqueValsForCatCols.append(numOfUniqueValsForCatCols[-1] + numOfUniqueVals)
                    hasMulticlassCat = True
            
            #Remove last index(no need)
            if(hasMulticlassCat): 
                del numOfUniqueValsForCatCols[-1]
                        
            oneHotEncoder = OneHotEncoder(categorical_features = feature_list, sparse=False)
            self.X = oneHotEncoder.fit_transform(self.X)
            
            '''Remove dummy variable'''
            self.X = np.delete(self.X, numOfUniqueValsForCatCols, 1)
        
     
    def NeuralNetwork(self):
        numOfCols = len(self.X[0])
        
        # Initialising the ANN
        classifier = Sequential()
        
        # Adding the input layer and the first hidden layer
        classifier.add(Dense(units = int(numOfCols/2), kernel_initializer = 'uniform', activation = 'relu', input_dim = numOfCols))
        
        # Adding the second hidden layer
        classifier.add(Dense(units = int(numOfCols/2), kernel_initializer = 'uniform', activation = 'relu'))
        
        # Adding the output layer
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        
        # Compiling the ANN
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        return classifier


    def RunEncapsulatedModel(self, classifier, modelType):
        if modelType == "Neural Network":
            classifier.fit(self.X_train, self.y_train, batch_size = 32, epochs = 50)
        else:
            classifier.fit(self.X_train, self.y_train)
    
        self.CheckMetrics(modelType, classifier)


    ''' Generates the model within the given parameters and run it '''
    def GenerateModel(self, params):
        modelType = params.get('modelType')
        
        if(modelType == "Logistic Regression"):
            classifier = LogisticRegression(random_state = 0)
                
        elif(modelType == "KNN"):
            classifier = KNeighborsClassifier(n_neighbors = params["numofneighbour"], metric = params["metric"], p = params["p"]) 
                
        elif(modelType == "Naive Bayes"):
            classifier = GaussianNB()
                
        elif(modelType == "Kernel SVM"):
            classifier = SVC(kernel = params["kernel"], random_state = 0)
            
        elif(modelType == "Decision Tree"):
            classifier = DecisionTreeClassifier(criterion = params["criterion"], random_state = 0)
            
        elif(modelType == "Random Forest"):
            classifier = RandomForestClassifier(n_estimators = params["estimators"], criterion = params["criterion"], random_state = 0)
            
        elif(modelType == "XGBoost"):
            classifier = XGBClassifier()
            
        elif(modelType == "Neural Network"):
            classifier = self.NeuralNetwork()
            
        #Run the classifier
        self.RunEncapsulatedModel(classifier, modelType)
        
    
    ''' Sends various parameters to GenerateModel method to create multiple models '''
    def ModelMultiplexer(self):
        #Logistic Regression Models - 1
        self.GenerateModel({"modelType": "Logistic Regression"})
            
        #KNN Models - 16
        for numofneighbour in range(3,7):
            for p in range(1,5):
                for metric in ["minkowski"]:
                    self.GenerateModel({"modelType": "KNN", "numofneighbour": numofneighbour, "metric": metric, "p": p})
                        
        #Naive Bayes Models - 1
        self.GenerateModel({"modelType": "Naive Bayes"})
            
        #Kernel SVM Models - 4
        for kernel in ["linear", "poly", "rbf", "sigmoid"]:
            self.GenerateModel({"modelType": "Kernel SVM", "kernel": kernel})
                
        #Decision Tree Models - 2
        for criterion in ["gini", "entropy"]:
            self.GenerateModel({"modelType": "Decision Tree", "criterion": criterion})
            
        #Random Forest Models - 14
        for estimators in range(8,14):
            for criterion in ["gini", "entropy"]:
                self.GenerateModel({"modelType": "Random Forest", "criterion": criterion, "estimators": estimators})
            
        #XGBoost Models - 1
        self.GenerateModel({"modelType": "XGBoost"})
            
        #Neural Network Models - 1
        self.GenerateModel({"modelType": "Neural Network"})
    
    
    
    
    def Run(self):
        try:
            ''' Save status '''
            self.SaveTrainStatus(0, 'Preprocess Starting...')
                
            #Preprocess dataset
            self.Preprocess()
            
            ''' Save status '''
            self.SaveTrainStatus(0, 'Preprocess Finished.GMS Starting...')
        
            #Create models, find best model
            if self.data.get('modelType'):
                self.GenerateModel(self.data)
            else:
                self.ModelMultiplexer()
                    
            ''' Save status '''
            self.SaveTrainStatus(0, 'GMS Finished.Best model is being saved.')
                
            #Save the best model
            self.SaveModel()
                    
            ''' Save status '''
            self.SaveTrainStatus(1, 'Best model is saved.')
                    
            print("GMS Finished Successfuly !")        
        except Exception as e:
            ''' Save status '''
            self.SaveTrainStatus(-1, 'GMS Finished with errors: ' + str(e))
            
            print("GMS Finished with errors: " + str(e))
            
            
