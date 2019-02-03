from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, Imputer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import keras
from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
import numpy as np

import pickle
import pymongo
import boto3


class GMS:
    
    maxScore = 0
    bestModel = ""
    userName = ""
    modelName = ""
    algorithm = ""
    dburi = "mongodb://webuser:789456123Aa.@cluster0-shard-00-00-l51oi.gcp.mongodb.net:27017,cluster0-shard-00-01-l51oi.gcp.mongodb.net:27017,cluster0-shard-00-02-l51oi.gcp.mongodb.net:27017/test?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin&retryWrites=true"
    
    
    def __init__(self, username, modelname, dataset, columns, target, categoricalcolumns, numericalcolumns):
        self.name = ''
        self.userName = username
        self.modelName = modelname
        self.dataset = dataset
        self.columns = columns
        self.target = target
        self.categoricalcolumns = categoricalcolumns
        self.numericalcolumns = numericalcolumns
        
        self.client = pymongo.MongoClient(self.dburi, ssl=True)
        self.db = self.client.churndb
        self.modeldetails = self.db.modeldetails
        self.trainstatus = self.db.trainstatus
        self.ss = StandardScaler()
    
    
    def SaveTrainStatus(self, status, detail):
        oldPost = {"username": self.userName, "modelname": self.modelName, "status": status, "detail": detail}
        self.trainstatus.update_one({"username": self.userName, "modelname": self.modelName}, {"$set": oldPost}, upsert=True)
    
    
    def SaveModel(self):
        '''Save best model to memory'''
        self.SaveModelToMemory()
        
        '''Save the bestModel path to database'''
        self.SaveModelToDB()
        
    
    def SaveModelToDB(self):
        oldPost = self.modeldetails.find_one({"username":  self.userName })
        
        catCols = []
        for catColIndex in self.categoricalcolumns:
            catCols.append(dict({ "name": self.columns[catColIndex], "values": sorted(pd.unique(self.data_frame.iloc[:, catColIndex].values).tolist()) }))
        
        numCols = []
        for numColIndex in self.numericalcolumns:
            numCols.append(self.columns[numColIndex])
            
        targetCol = dict({ "name": self.columns[self.target], "values": sorted(pd.unique(self.data_frame.iloc[:, self.target].values).tolist()) })
        
        newModel = {"modelname": self.modelName, "catCols": catCols , "numCols": numCols, "targetCol": targetCol, "algorithm": self.algorithm, "accuracy": self.maxScore}
        
        if oldPost is None:
            '''User doesn't have any model previously '''
            oldPost = {"username": self.userName, "models": [dict(newModel)]}
        else:
            '''User has at least one model before '''
            prevModels = oldPost["models"]
            prevModels.append(dict(newModel))
            oldPost["models"] = prevModels
        
        
        self.modeldetails.update_one({"username": self.userName}, {"$set": dict(oldPost)}, upsert=True)
        
    

    def SaveModelToMemory(self):
        modelPath = self.userName + self.modelName + ".txt"
        scalerPath = self.userName + self.modelName + "scaler.txt"
        s3 = boto3.resource('s3')
        
        bucket_name = 'churn-bucket'
        
        modelInBytes = pickle.dumps(self.bestModel)
        scalerInBytes = pickle.dumps(self.ss)
 
        s3.meta.client.put_object(Body=modelInBytes, Bucket=bucket_name, Key=modelPath)
        s3.meta.client.put_object(Body=scalerInBytes, Bucket=bucket_name, Key=scalerPath)
        


    def Preprocess(self):        
        ''' Make dataset a dataframe '''
        self.data_frame = pd.DataFrame(self.dataset, columns = self.columns)
        self.data_frame = self.data_frame[self.columns].apply(pd.to_numeric, errors="ignore")

        #categoricalRange = list(range(0, len(self.categoricalcolumns)))

        print(self.data_frame)
        
        ''' Handle Missing Values in categorical columns '''
        for col in self.categoricalcolumns:
            print(self.data_frame.iloc[:, col].value_counts())
            self.data_frame.fillna(self.data_frame.iloc[:, col].value_counts().index[0], axis = 1, inplace = True)
        
        ''' Assign columns'''
        self.X = self.data_frame.iloc[:, (self.categoricalcolumns + self.numericalcolumns)].values
        self.y = self.data_frame.iloc[:, self.target].values
        
        numericalRange = list(range(len(self.categoricalcolumns), len(self.categoricalcolumns) + len(self.numericalcolumns)))
        
        print(self.X)
        
        ''' Handle Missing Values in numerical columns '''
        imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
        imputer.fit(self.X[:, numericalRange])
        self.X[:, numericalRange] = imputer.transform(self.X[:, numericalRange])
        
        print(self.X)
        
        ''' Encode categorical vars '''
        self.EncodeCategoricalVars()
        
        ''' Encode y column '''
        labelEncoder = LabelEncoder()
        self.y = labelEncoder.fit_transform(self.y)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = .25, random_state = 0)
        
        ''' Scale vars '''
        self.X_train = self.ss.fit_transform(self.X_train)
        self.X_test = self.ss.transform(self.X_test)
    
    
    '''Encoding categorical data'''
    def EncodeCategoricalVars(self):
        feature_list = []
        numOfUniqueValsForCatCols = []
        hasMulticlassCat = False
        indexZeroAdded = False
        
        #Check if there is categorical variable
        if len(self.categoricalcolumns) != 0:
            i = 0
            for col in self.categoricalcolumns:
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
                i += 1
            
            #Remove last index(no need)
            if(hasMulticlassCat): 
                del numOfUniqueValsForCatCols[-1]
            
            oneHotEncoder = OneHotEncoder(categorical_features = feature_list)
            self.X = oneHotEncoder.fit_transform(self.X).toarray()
            
            '''Remove dummy variable'''
            self.X = np.delete(self.X, numOfUniqueValsForCatCols, 1)
        
    
    
    def LogisticRegression(self):
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        
        y_train_predict = classifier.predict(self.X_train)
        y_test_predict = classifier.predict(self.X_test)
        
        accuracy_train = accuracy_score(self.y_train, y_train_predict)
        accuracy_test = accuracy_score(self.y_test, y_test_predict)
        
        print("Logistic Regression Train Accuracy:")
        print(accuracy_train)
        print("Logistic Regression Test Accuracy:")
        print(accuracy_test)
        
        if(abs(accuracy_test - accuracy_train) <= 6):
            if(((accuracy_train + accuracy_test) / 2) > self.maxScore):
                self.bestModel = classifier
                self.maxScore = (accuracy_train + accuracy_test) / 2
                self.algorithm = "Logistic Regression"
            
        
    def KNN(self, pnumofneighbour, pmetric, pp):
        classifier = KNeighborsClassifier(n_neighbors = pnumofneighbour, metric = pmetric, p = pp)
        classifier.fit(self.X_train, self.y_train)
        
        y_train_predict = classifier.predict(self.X_train)
        y_test_predict = classifier.predict(self.X_test)
        
        accuracy_train = accuracy_score(self.y_train, y_train_predict)
        accuracy_test = accuracy_score(self.y_test, y_test_predict)
        
        print("KNN Train Accuracy:")
        print(accuracy_train)
        print("KNN Test Accuracy:")
        print(accuracy_test)
        
        if(abs(accuracy_test - accuracy_train) <= 6):
            if(((accuracy_train + accuracy_test) / 2) > self.maxScore):
                self.bestModel = classifier
                self.maxScore = (accuracy_train + accuracy_test) / 2
                self.algorithm = "KNN"
       
        
    def NaiveBayes(self):
        classifier = GaussianNB()
        classifier.fit(self.X_train, self.y_train)
        
        y_train_predict = classifier.predict(self.X_train)
        y_test_predict = classifier.predict(self.X_test)
        
        accuracy_train = accuracy_score(self.y_train, y_train_predict)
        accuracy_test = accuracy_score(self.y_test, y_test_predict)
        
        print("Naive Bayes Train Accuracy:")
        print(accuracy_train)
        print("Naive Bayes Test Accuracy:")
        print(accuracy_test)
        
        if(abs(accuracy_test - accuracy_train) <= 6):
            if(((accuracy_train + accuracy_test) / 2) > self.maxScore):
                self.bestModel = classifier
                self.maxScore = (accuracy_train + accuracy_test) / 2
                self.algorithm = "Naive Bayes"
        
        
    def KernelSVM(self, pkernel):
        classifier = SVC(kernel = pkernel, random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        
        y_train_predict = classifier.predict(self.X_train)
        y_test_predict = classifier.predict(self.X_test)
        
        accuracy_train = accuracy_score(self.y_train, y_train_predict)
        accuracy_test = accuracy_score(self.y_test, y_test_predict)
        
        print("Kernel SVM Train Accuracy:")
        print(accuracy_train)
        print("Kernel SVM Test Accuracy:")
        print(accuracy_test)
        
        if(abs(accuracy_test - accuracy_train) <= 6):
            if(((accuracy_train + accuracy_test) / 2) > self.maxScore):
                self.bestModel = classifier
                self.maxScore = (accuracy_train + accuracy_test) / 2
                self.algorithm = "Kernel SVM"
            

    def DecisionTree(self, pcriterion):
        classifier = DecisionTreeClassifier(criterion = pcriterion, random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        
        y_train_predict = classifier.predict(self.X_train)
        y_test_predict = classifier.predict(self.X_test)
        
        accuracy_train = accuracy_score(self.y_train, y_train_predict)
        accuracy_test = accuracy_score(self.y_test, y_test_predict)
        
        print("Decision Tree Train Accuracy:")
        print(accuracy_train)
        print("Decision Tree Test Accuracy:")
        print(accuracy_test)
        
        if(abs(accuracy_test - accuracy_train) <= 6):
            if(((accuracy_train + accuracy_test) / 2) > self.maxScore):
                self.bestModel = classifier
                self.maxScore = (accuracy_train + accuracy_test) / 2
                self.algorithm = "Decision Tree"


    def RandomForest(self, pestimators, pcriterion):
        classifier = RandomForestClassifier(n_estimators = pestimators, criterion = pcriterion, random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        
        y_train_predict = classifier.predict(self.X_train)
        y_test_predict = classifier.predict(self.X_test)
        
        accuracy_train = accuracy_score(self.y_train, y_train_predict)
        accuracy_test = accuracy_score(self.y_test, y_test_predict)
        
        print("Random Forest Train Accuracy:")
        print(accuracy_train)
        print("Random Forest Test Accuracy:")
        print(accuracy_test)
        
        if(abs(accuracy_test - accuracy_train) <= 6):
            if(((accuracy_train + accuracy_test) / 2) > self.maxScore):
                self.bestModel = classifier
                self.maxScore = (accuracy_train + accuracy_test) / 2
                self.algorithm = "Random Forest"
            
            
    def ArtificialNeuralNetwork(self):
        
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
        
        # Fitting the ANN to the Training set
        classifier.fit(self.X_train, self.y_train, batch_size = 32, epochs = 50)
        
        # Predicting the Test set results
        y_train_predict = classifier.predict(self.X_train)
        y_train_predict = (y_train_predict > 0.5)
        
        y_test_predict = classifier.predict(self.X_test)
        y_test_predict = (y_test_predict > 0.5)
        
        accuracy_train = accuracy_score(self.y_train, y_train_predict)
        accuracy_test = accuracy_score(self.y_test, y_test_predict)
        
        print("Neural Network Train Accuracy:")
        print(accuracy_train)
        print("Neural Network Test Accuracy:")
        print(accuracy_test)
        
        if(abs(accuracy_test - accuracy_train) <= 6):
            if(((accuracy_train + accuracy_test) / 2) > self.maxScore):
                self.bestModel = classifier
                self.maxScore = (accuracy_train + accuracy_test) / 2
                self.algorithm = "Neural Network"


    '''Generates given model type with different parameters and assigns highest acc. model'''
    def GenerateModels(self, modelType):        
        '''General -> Train-Test Size'''
        '''KNN -> neighbournum, metric, p'''
        '''KernelSVM -> kernel'''
        '''DecisionTree -> criterion'''
        '''RandomForest -> estimators, criterion'''
        if(modelType == "LogisticRegression"):
            self.LogisticRegression()
        elif(modelType == "KNN"):
            for numofneighbour in range(3,7):
                for p in range(1,5):
                    for metric in ["minkowski"]:
                        self.KNN(numofneighbour, metric, p)        
        elif(modelType == "NaiveBayes"):
            self.NaiveBayes()
        elif(modelType == "KernelSVM"):
            for kernel in ["linear", "poly", "rbf", "sigmoid"]:
                self.KernelSVM(kernel)
        elif(modelType == "DecisionTree"):
            for criterion in ["gini", "entropy"]:
                self.DecisionTree(criterion)
        elif(modelType == "RandomForest"):
            for estimators in range(8,14):
                for criterion in ["gini", "entropy"]:
                    self.RandomForest(estimators, criterion)
        elif(modelType == "ArtificialNeuralNetwork"):
            self.ArtificialNeuralNetwork()
    
    
    def Run(self):
        
        ''' Save status '''
        self.SaveTrainStatus(0, 'Preprocess Starting...')
            
        '''Preprocess dataset'''
        self.Preprocess()
            
        ''' Save status '''
        self.SaveTrainStatus(0, 'Preprocess Finished.')
            
        ''' Save status '''
        self.SaveTrainStatus(0, 'GMS Starting...')

        '''Create models, find best model'''
        self.GenerateModels("LogisticRegression")
        self.GenerateModels("KNN")
        self.GenerateModels("NaiveBayes")
        self.GenerateModels("KernelSVM")
        self.GenerateModels("DecisionTree")
        self.GenerateModels("RandomForest")
        self.GenerateModels("ArtificialNeuralNetwork")
            
        ''' Save status '''
        self.SaveTrainStatus(0, 'GMS Finished.')
        
        ''' Save status '''
        self.SaveTrainStatus(0, 'Best model is being saved.')
        
        ''' Save the best model '''
        self.SaveModel()
            
        ''' Save status '''
        self.SaveTrainStatus(1, 'Best model is saved.')
            
        print("GMS Finished Successfuly !")
        """
        try:
            ''' Save status '''
            self.SaveTrainStatus(0, 'Preprocess Starting...')
            
            '''Preprocess dataset'''
            self.Preprocess()
            
            ''' Save status '''
            self.SaveTrainStatus(0, 'Preprocess Finished.')
            
            ''' Save status '''
            self.SaveTrainStatus(0, 'GMS Starting...')

            '''Create models, find best model'''
            self.GenerateModels("LogisticRegression")
            self.GenerateModels("KNN")
            self.GenerateModels("NaiveBayes")
            self.GenerateModels("KernelSVM")
            self.GenerateModels("DecisionTree")
            self.GenerateModels("RandomForest")
            self.GenerateModels("ArtificialNeuralNetwork")
            
            ''' Save status '''
            self.SaveTrainStatus(0, 'GMS Finished.')
            
            ''' Save status '''
            self.SaveTrainStatus(0, 'Best model is being saved.')
            
            ''' Save the best model '''
            self.SaveModel()
            
            ''' Save status '''
            self.SaveTrainStatus(1, 'Best model is saved.')
            
            print("GMS Finished Successfuly !")
        except Exception as e:
            ''' Save status '''
            self.SaveTrainStatus(-1, 'GMS Finished with errors: ' + str(e))
            
            print("GMS Finished with errors: " + str(e))
            
            """
            
