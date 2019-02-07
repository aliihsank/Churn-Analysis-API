from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api

from sklearn.preprocessing import StandardScaler

from threading import Thread

import pandas as pd
import numpy as np

from datetime import datetime, timedelta

import pickle
import pymongo
import boto3

from GMS import GMS


dburi = "mongodb://webuser:789456123Aa.@cluster0-shard-00-00-l51oi.gcp.mongodb.net:27017,cluster0-shard-00-01-l51oi.gcp.mongodb.net:27017,cluster0-shard-00-02-l51oi.gcp.mongodb.net:27017/test?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin&retryWrites=true"


app = Flask(__name__)
api = Api(app)
CORS(app)


def MakeValidations(username, password, requestedMethod):
    if (ValidateUser(username, password)):
        if(ValidateUserPlan(username, requestedMethod)):
            return True
        
    return False


def ValidateUser(username, password):
    client = pymongo.MongoClient(dburi, ssl=True)
    db = client.churndb
            
    if db.userdetails.find_one({"username": username, "password": password}) is None:
        return False
    else:
        return True
    
    
def ValidateUserPlan(username, requestedMethod):
    
    """
    User Attributes:
        -User Type
        -columnsInfos
        -train
        -predict
        
    Beginner:
        -Free
        -checkTrainStatus : Unlimited
        -removeModel : Unlimited
        -columnsInfos : 2
        -train : 2
        -modellist : Unlimited
        -predict : 10 (Single or Multiple)
        
    Hobbiest:
        -X Dollar
        -checkTrainStatus : Unlimited
        -removeModel : Unlimited
        -columnsInfos : 5
        -train : 5
        -modellist : Unlimited
        -predict : 20 (Single or Multiple)
    
    
    Professional:
        -2X Dollar
        -checkTrainStatus : Unlimited
        -removeModel : Unlimited
        -columnsInfos : 10
        -train : 10
        -modellist : Unlimited
        -predict : 40 (Single or Multiple)
    
    """
    
    client = pymongo.MongoClient(dburi, ssl=True)
    db = client.churndb
    
    oldPost = db.userdetails.find_one({"username": username})
    
    try:
        if oldPost["endDate"] < datetime.now():
            if oldPost[requestedMethod] > 0:
                oldPost[requestedMethod] -= 1
                db.userdetails.update_one({"username": username}, {"$set": dict(oldPost)}, upsert=True)
                return True
            else:
                return False
        else:
            return False
    except Exception as e:
        return False
        
    
        
def LoadScalerFrom(scalerPath):
    s3 = boto3.resource("s3").Bucket("churn-bucket")
    scaler = pickle.loads(s3.Object(key=scalerPath).get()["Body"].read())
    
    return scaler


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
        
        usertype = "Beginner"
        columnsInfos = 2
        train = 2
        predict = 10
        
        if email is None or username is None or password is None:
            return {'info': '0'}
        else:
            try:
                client = pymongo.MongoClient(dburi, ssl=True)
                db = client.churndb
                
                if db.userdetails.find_one({"username": username}) is None:
                    post = { "email":email, "username": username, "password":password, "usertype": usertype, "columnsInfos": columnsInfos, "train": train, "predict": predict, "endDate": (datetime.now() + timedelta(days=365)) }
                    db.userdetails.insert_one(post)
                    return {'info': 1}
                else:
                    return {'info': 0}
            except Exception as e:
                return {'info': -1, 'details': str(e)}
    


class UpdateUserPlan(Resource):
    def post(self):
        data = request.get_json()
        
        username = data["username"]
        password = data["password"]
        usertype = data["usertype"]
        
        if usertype == "Hobbiest":
            columnsInfos = 5
            train = 5
            predict = 20
        elif usertype == "Professional":
            columnsInfos = 10
            train = 10
            predict = 40
        else:
            return {'info': 0}
            
        try:
            client = pymongo.MongoClient(dburi, ssl=True)
            db = client.churndb
                
            oldPost = db.userdetails.find_one({"username": username, "password": password})
            oldPost["usertype"] = usertype
            oldPost["columnsInfos"] = columnsInfos
            oldPost["train"] = train
            oldPost["predict"] = predict
            oldPost["endDate"] = (datetime.now() + timedelta(days=365))
            
            db.userdetails.update_one({"username": username}, {"$set": dict(oldPost)}, upsert=True)
            return {'info': 1}
        except Exception as e:
            return {'info': -1, 'details': str(e)}
    
    
    
class Login(Resource):
    def post(self):
        data = request.get_json()
        
        username = data["username"]
        password = data["password"]
        
        try:
            client = pymongo.MongoClient(dburi, ssl=True)
            db = client.churndb
            
            if db.userdetails.find_one({"username": username, "password": password}) is None:
                return {'info': '0'}
            else:
                return {'info': '1'}
        except Exception as e:
            return {'info': -1, 'details': str(e)}


class GetUserPlan(Resource):
    def post(self):
        data = request.get_json()
        
        username = data["username"]
        password = data["password"]
        
        try:
            client = pymongo.MongoClient(dburi, ssl=True)
            db = client.churndb
            
            userdetails = db.userdetails.find_one({"username": username, "password": password})
            
            return {'info': 1, 'user': userdetails}
            
        except Exception as e:
            return {'info': -1, 'details': str(e)}



class CheckTrainStatus(Resource):
    def post(self):
        data = request.get_json()

        username = data["username"]
        password = data["password"]
        
        try:
            client = pymongo.MongoClient(dburi, ssl=True)
            db = client.churndb
            
            if(MakeValidations(username, password, 'checkTrainStatus')):
                statuslistCursor = db.trainstatus.find({"username": username}, {'_id': 0})
                if statuslistCursor is None:
                    return {'info': 0}
                else:
                    statuslist = []
                    for status in statuslistCursor:
                        statuslist.append(status)
                    return {'info': 1, 'statuslist': statuslist }
            else:
                return {'info': 0}
        except Exception as e:
            return {'info': -1, 'details': str(e)}
            


class RemoveModel(Resource):
    def post(self):
        data = request.get_json()
        
        username = data["username"]
        password = data["password"]
        modelname = data["modelname"]
        
        try:
            client = pymongo.MongoClient(dburi, ssl=True)
            db = client.churndb
            
            if(MakeValidations(username, password, 'removeModel')):
                client = pymongo.MongoClient(dburi, ssl=True)
                db = client.churndb
        
                #Remove from modeldetails
                oldPost = db.modeldetails.find_one({"username": username })
        
                '''User has at least one model before '''
                prevModels = oldPost["models"]
                newModels = []
                inModelDetails = False
                for model in prevModels:
                    if model["modelname"] == modelname:
                        inModelDetails = True
                    else:
                        newModels.append(dict(model))
                if inModelDetails:
                    oldPost["models"] = newModels
                    db.modeldetails.update_one({"username": username}, {"$set": dict(oldPost)}, upsert=True)
                    return {'info': 1, 'details': 'Removed from Trained Models'}
                else:
                    #Remove from trainstatus
                    db.trainstatus.delete_one({"username": username, "modelname": modelname })
                    return {'info': 1, 'details': 'Removed from StatusList'}
                
        except Exception as e:
            return {'info': -1, 'details': str(e)}
                


class ColumnsInfos(Resource):
    def post(self):
        data = request.get_json()
        
        username = data["username"]
        password = data["password"]
        columns = data["columns"]
        dataset = data["dataset"]
        
        try:
            if(MakeValidations(username, password, 'columnsInfos')):
                data_frame = pd.DataFrame(dataset, columns = columns)
                data_frame = data_frame[columns].apply(pd.to_numeric, errors="ignore")
                
                colInfos = []
                
                for i in range(len(columns)):
                    cat = 0
                    if(data_frame.iloc[:,[i]].values.dtype is np.dtype("object")):
                        cat = 1
                    
                    counterObj = data_frame.iloc[:, i].value_counts()
                    
                    colInfos.append({"name": columns[i], "values": counterObj.keys().tolist(), "counts": counterObj.tolist(), "cat": cat })
                    
                return {'info': 1, 'colInfos': colInfos}
            else:
                return {'info': 0}
        except Exception as e:
            return {'info': -1, 'details': str(e)}
     


class Train(Resource):
    def post(self):
        data = request.get_json()
        
        username = data["username"]
        password = data["password"]
        modelname = data["modelname"]
        isCustomized = data["isCustomized"]
        
        try:
            if(MakeValidations(username, password, 'train')):
                client = pymongo.MongoClient(dburi, ssl=True)
                db = client.churndb
                
                modelnameExists = False
                usermodelsInfo = db.modeldetails.find_one({"username": username })
                if usermodelsInfo is not None:
                    for model in usermodelsInfo["models"]:
                        if(modelname == model["modelname"]):
                            modelnameExists = True
                usermodelsInfo = db.trainstatus.find_one({"username": username, "modelname": modelname })
                if usermodelsInfo is not None:
                    if usermodelsInfo["status"] == 0:
                        return {'info': 0}
                if modelnameExists:
                    return {'info': 0}
                else:
                    gms = GMS(data)
                    
                    run = Thread(target = gms.Run, args = (isCustomized))
                    run.start()
                    return {'info': 1}
            else:
                return {'info': 0}
        except Exception as e:
            return {'info': -1, 'details': str(e)}
        
        
class ModelList(Resource):
    def post(self):
        data = request.get_json()
        
        username = data["username"]
        password = data["password"]
        
        try:
            if(MakeValidations(username, password, 'modellist')):
                client = pymongo.MongoClient(dburi, ssl=True)
                db = client.churndb
                
                if db.modeldetails.find_one({"username": username}, {'_id': 0}) is None:
                    return {"info": 0}
                else:
                    post = db.modeldetails.find_one({"username": username}, {'_id': 0})
                    return {"info": 1, "models": post["models"]}
            else:
                return {"info": 0}
        except Exception as e:
            return {'info': -1, 'details': str(e)}        

        
class Predict(Resource):
    def post(self):
        data = request.get_json()
        
        username = data["username"]
        password = data["password"]
        modelname = data["modelname"]
        predictset = data["predictset"]
        
        try:
            #Load Model
            if(MakeValidations(username, password, 'predict')):
                #Load Model
                model = LoadModelFrom(username + modelname + ".txt")
            
                #Feature Scaling (predictset comes onehotencoded)
                ss = LoadScalerFrom(username + modelname + "scaler.txt")
                predictset = ss.transform(predictset)
                
                #Make prediction
                result = model.predict(predictset).tolist()
                #Return result
                return {'info': 1, 'prediction': result}
            else:
                return {"info": 0}
        except Exception as e:
            return {'info': -1, 'details': str(e)}
        
        
        
api.add_resource(ColumnsInfos, '/columnsInfos')
api.add_resource(Train, '/train')
api.add_resource(Test, '/test')
api.add_resource(Register, '/register')
api.add_resource(Login, '/login')
api.add_resource(ModelList, '/modelList')
api.add_resource(Predict, '/predict')
api.add_resource(CheckTrainStatus, '/checkStatus')
api.add_resource(RemoveModel, '/removeModel')
api.add_resource(GetUserPlan, '/getUserPlan')
api.add_resource(UpdateUserPlan, '/updateUserPlan')

if __name__ == '__main__':
    app.run(debug = False)
    
    
