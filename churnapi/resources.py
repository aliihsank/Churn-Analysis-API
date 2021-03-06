from churnapi import api, db
from flask import request
from flask_restful import Resource

from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import json

from threading import Thread
import pickle

import firebase_admin
from firebase_admin import storage


from .GMS import GMS


def MakeValidations(uid, requestedMethod):
    if (ValidateUser(uid)):
        if(ValidateUserPlan(uid, requestedMethod)):
            return True
        
    return False


def ValidateUser(uid):
    user_ref = db.collection(u'users').document(u'' + uid)
    
    if user_ref.get().to_dict() is None:
        return False
    else:
        return True
    
    
def ValidateUserPlan(uid, requestedMethod):
    
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
        
    Hobbyist:
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
    if requestedMethod == "columnsInfos" or requestedMethod == "train" or requestedMethod == "predict":
        
        
        user_ref = db.collection(u'users').document(u'' + uid)
    
        oldPost = user_ref.get().to_dict()
        try:
            if oldPost[requestedMethod] > 0:
                oldPost[requestedMethod] -= 1
                db.collection(u'users').document(u'' + uid).set(dict(oldPost))
                return True
            else:
                return False
        except Exception as e:
            return False
    else:
        return True
        
    
        
def LoadScalerFrom(scalerPath):        
    default_bucket = storage.bucket(name="churn-2537f.appspot.com", app=None)
    
    scalerBlob = default_bucket.blob(scalerPath)
    loaded_scaler = pickle.loads(scalerBlob.download_as_string())
    
    return loaded_scaler


def LoadModelFrom(modelPath):
    default_bucket = storage.bucket(name="churn-2537f.appspot.com", app=None)
    
    modelBlob = default_bucket.blob(modelPath)
    loaded_model = pickle.loads(modelBlob.download_as_string())
    
    return loaded_model
    


class MainPage(Resource):
    def get(self):
        
        author = "Ali İhsan Karabal"
        definition = "This is Churn Analysis Project's Backend"
        methods = [{"Method": "GET", "URL": "/"}, {"Method": "GET", "URL": "/test"}]
        
        return {"author": author, "definition": definition, "methods": methods}
            

class Test(Resource):
    def get(self):
        return {'Test Message': "Hello User"}
    

class ColumnsInfos(Resource):
    def post(self):
        data = request.get_json()
        
        uid = data["uid"]
        columns = data["columns"]
        dataset = data["dataset"]
        
        try:
            if(MakeValidations(uid, 'columnsInfos')):
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
        
        uid = data["uid"]
        modelname = data["modelname"]
        
        try:
            if(MakeValidations(uid, 'train')):
                doc = db.collection(u'models').document(u'' + uid)
    
                modelnameExists = False
                usermodelsInfo = doc.get().to_dict()
                if usermodelsInfo is not None:
                    for model in usermodelsInfo["models"]:
                        if(modelname == model["modelname"]):
                            modelnameExists = True
                doc = db.collection(u'trainstatus').document(u'' + uid)
                usermodelsInfo = doc.get().to_dict()
                if usermodelsInfo is not None:
                    if usermodelsInfo["status"] == 0:
                        return {'info': 0, 'details': 'This model name already exists. Please enter another name.'}
                if modelnameExists:
                    return {'info': 0, 'details': 'This model name already exists. Please enter another name.'}
                else:
                    gms = GMS(data)
                    #gms.Run()
                    run = Thread(target = gms.Run, args = ())
                    run.start()
                    return {'info': 1}
            else:
                return {'info': 0, 'details': 'Your have reached your limit.'}
        except Exception as e:
            print("Error:" + str(e))
            return {'info': -1, 'details': str(e)}
            

        
class Predict(Resource):
    def post(self):
        data = request.get_json()
        
        uid = data["uid"]
        modelname = data["modelname"]
        predictset = data["predictset"]
        
        try:
            #Load Model
            if(MakeValidations(uid, 'predict')):
                #Load Model
                model = LoadModelFrom(uid + modelname + ".txt")
            
                #Feature Scaling (predictset comes onehotencoded)
                ss = LoadScalerFrom(uid + modelname + "scaler.txt")
                predictset = ss.transform(predictset)
                
                #Make prediction
                ''' Formatted to return true results for NN too '''
                result = [int(i > 0.5) for i in model.predict(predictset)]
                
                #Return result
                return {'info': 1, 'prediction': result}
            else:
                return {"info": 0}
        except Exception as e:
            print("Error: " + str(e))
            return {'info': -1, 'details': str(e)}

     
        
api.add_resource(MainPage, '/')
api.add_resource(Test, '/test')

api.add_resource(ColumnsInfos, '/columnsInfos')
api.add_resource(Train, '/train')
api.add_resource(Predict, '/predict')

