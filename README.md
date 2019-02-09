# Project : Churnify

- This project is in development
- This project is a part of my graduation - thesis project

-----------------------------

- This project is being developed using flask-restful, pymongo, boto3, scikit-learn, tensorflow and keras
- Project uses a NoQSL MongoDB to store users and models details
- This project runs 6 different classification algorithm from scikit-learn and neural network with different parameters from keras


## USED SERVICES ##
- Heroku to publish python-flask server app
- MongoDB cloud for nosql database
- AWS S3 for file storage
- Keras for neural network construction
- Tensorflow as backend to Keras

## METHODS ##

It has following methods that can be requested:

- MainPage ()
- Test ()           

- Register (email, username, password)
- Login (username, password)

- GetUserPlan (username, password)
- UpdateUserPlan (username, password, usertype)

- ColumnsInfos (username, password, columns, dataset)
- Train (username, password, modelname, dataset, columns, target, categoricalcolumns, numericalcolumns, modelType = None, {classifier specific variables})
- Predict (username, password, modelname, predictset)

- ModelList(username, password)
- CheckTrainStatus (username, password)
- RemoveModel (username, password, modelname)

## GMS Module Details ##

Also there is GMS(Generative Model Selector) class working in background to find best algorithm and parameters for given dataset

GMS has 6 different classification algorithm and neural network to try with different parameters:
- Logistic Regression
- KNN
- Naive Bayes
- Kernel SVM
- Decision Tree
- Random Forest
- Neural Network

Live version of this code is in:

https://churn-analysis-api.herokuapp.com/


You can send requests to following URLs:

GET https://churn-analysis-api.herokuapp.com/

GET https://churn-analysis-api.herokuapp.com/test


POST https://churn-analysis-api.herokuapp.com/register

POST https://churn-analysis-api.herokuapp.com/login


POST https://churn-analysis-api.herokuapp.com/getUserPlan

POST https://churn-analysis-api.herokuapp.com/updateUserPlan


POST https://churn-analysis-api.herokuapp.com/columnsInfos

POST https://churn-analysis-api.herokuapp.com/train

POST https://churn-analysis-api.herokuapp.com/predict


POST https://churn-analysis-api.herokuapp.com/modelList

POST https://churn-analysis-api.herokuapp.com/checkStatus

POST https://churn-analysis-api.herokuapp.com/removeModel



### TODOs: ###
- Convolutional Neural Network Option
- Optimization
- Encapsulation
