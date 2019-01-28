from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api

from sklearn.preprocessing import StandardScaler

from threading import Thread

import pandas as pd
import numpy as np

import pickle
import pymongo
import boto3

from GMS import GMS


data_frame = []

app = Flask(__name__)
api = Api(app)
CORS(app)


def ValidateUser(username, password):
    uri = "mongodb://webuser:789456123Aa.@cluster0-shard-00-00-l51oi.gcp.mongodb.net:27017,cluster0-shard-00-01-l51oi.gcp.mongodb.net:27017,cluster0-shard-00-02-l51oi.gcp.mongodb.net:27017/test?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin&retryWrites=true"
    client = pymongo.MongoClient(uri, ssl=True)
    db = client.churndb
            
    if db.userdetails.find_one({"username": username, "password": password}) is None:
        return False
    else:
        return True
        
        
def LoadModelFrom(modelPath):
    s3 = boto3.resource("s3").Bucket("churn-bucket")
    
    loaded_model = pickle.loads(s3.Object(key=modelPath).get()["Body"].read())
    
    return loaded_model
    

class Test(Resource):
    def get(self):
        return {'Test Message': "Hello User"}
    

class Register(Resource):
    def post(self):
        data = request.get_json()
        
        email = data["email"]
        username = data["username"]
        password = data["password"]
        
        if email is None or username is None or password is None:
            return {'info': '0'}
        else:
            uri = "mongodb://webuser:789456123Aa.@cluster0-shard-00-00-l51oi.gcp.mongodb.net:27017,cluster0-shard-00-01-l51oi.gcp.mongodb.net:27017,cluster0-shard-00-02-l51oi.gcp.mongodb.net:27017/test?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin&retryWrites=true"
            client = pymongo.MongoClient(uri, ssl=True)
            db = client.churndb
            
            post = { "email":email, "username": username, "password":password }
            db.userdetails.insert_one(post)
            return {'info': '1'}
    
    
class Login(Resource):
    def post(self):
        data = request.get_json()
        
        username = data["username"]
        password = data["password"]

        uri = "mongodb://webuser:789456123Aa.@cluster0-shard-00-00-l51oi.gcp.mongodb.net:27017,cluster0-shard-00-01-l51oi.gcp.mongodb.net:27017,cluster0-shard-00-02-l51oi.gcp.mongodb.net:27017/test?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin&retryWrites=true"
        client = pymongo.MongoClient(uri, ssl=True)
        db = client.churndb
        
        if db.userdetails.find_one({"username": username, "password": password}) is None:
            return {'info': '0'}
        else:
            return {'info': '1'}
    


class ColumnsInfos(Resource):
    def post(self):
        data = request.get_json()
        
        username = data["username"]
        password = data["password"]
        columns = data["columns"]
        dataset = data["dataset"]
        
        if(ValidateUser(username, password)):
            data_frame = pd.DataFrame(dataset, columns = columns)
            data_frame = data_frame[columns].apply(pd.to_numeric, errors="ignore")
            
            result = []
            
            for i in range(len(columns)):
                cat = 0
                if(data_frame.iloc[:,[i]].values.dtype is np.dtype("object")):
                    cat = 1
                num = len(set(data_frame.iloc[:,i]))
                
                result.append({"name": columns[i] , "number": num, "cat": cat })
            
            return {'colInfos': result}
        else:
            return {'error': 'User is not registered !'}
        

class Train(Resource):
    def post(self):
        data = request.get_json()
        
        username = data["username"]
        password = data["password"]
        modelname = data["modelname"]
        dataset = data["dataset"]
        columns = data["columns"]
        target = data["target"]
        categoricalcolumns = data["categoricalcolumns"]
        numericalcolumns = data["numericalcolumns"]
        
        if(ValidateUser(username, password)):
            try:
                gms = GMS(username, modelname, dataset, columns, target, categoricalcolumns, numericalcolumns)
                
                run = Thread(target = gms.Run, args = ())
                run.start()
            except Exception as e:
                print("Hata Oluştu !!" + str(e))
                return {'error': 'An error occured !! ' + str(e)}
            
            return {'info': 'training started !'}
        else:
            return {'error': 'User is not registered !'}
        
        
class ModelList(Resource):
    def post(self):
        data = request.get_json()
        
        username = data["username"]
        password = data["password"]
        
        if(ValidateUser(username, password)):
            uri = "mongodb://webuser:789456123Aa.@cluster0-shard-00-00-l51oi.gcp.mongodb.net:27017,cluster0-shard-00-01-l51oi.gcp.mongodb.net:27017,cluster0-shard-00-02-l51oi.gcp.mongodb.net:27017/test?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin&retryWrites=true"
            client = pymongo.MongoClient(uri, ssl=True)
            db = client.churndb
            
            if db.modeldetails.find_one({"username": username}, {'_id': 0}) is None:
                return {"error": "User doesn't have any model"}
            else:
                post = db.modeldetails.find_one({"username": username}, {'_id': 0})
                return post["models"]
        else:
            return {"error": "User doesn't exist"}
        
        
class Predict(Resource):
    def post(self):
        data = request.get_json()
        
        username = data["username"]
        password = data["password"]
        modelname = data["modelname"]
        predictset = data["predictset"]
        
        if(ValidateUser(username, password)):
            #Load Model
            model = LoadModelFrom(username + modelname + ".txt")
            
            #Feature Scaling
            ss = StandardScaler()
            predictset = ss.fit_transform(predictset)
            
            #Make prediction
            result = model.predict(predictset)
            
            #Return result
            return {'prediction': result}
            
        else:
            return {"error": "User doesn't exist"}
            
        
        
        
api.add_resource(ColumnsInfos, '/columnsInfos')
api.add_resource(Train, '/train')
api.add_resource(Test, '/test')
api.add_resource(Register, '/register')
api.add_resource(Login, '/login')
api.add_resource(ModelList, '/modelList')
api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run()
    
    
