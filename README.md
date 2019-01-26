# Project : Churnify

- This project is in alpha
- This project is a part of my graduation - thesis project

-----------------------------

- This project is being developed using flask-restful, pymongo, scikit-learn and keras
- Project uses a NoQSL MongoDB to store users and models details
- This project runs 6 different classification algorithm from scikit-learn and neural network with different parameters from keras

-------------------------------

It has following methods that can be requested:

- Test ()           
- Register (email, username, password)
- Login (username, password)
- ColumnsInfos (username, password, columns, dataset)
- Train (username, password, modelname, dataset, columns, target, categoricalcolumns, numericalcolumns)


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

https://denemeq.herokuapp.com/

You can send requests to following URLs:

GET https://denemeq.herokuapp.com/test


POST https://denemeq.herokuapp.com/register

POST https://denemeq.herokuapp.com/login


POST https://denemeq.herokuapp.com/columnsInfos

POST https://denemeq.herokuapp.com/train
