from churnapi import api, db
from flask import request
from flask_restful import Resource

from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import json

import pickle
import boto3

from .GMS import GMS


def MakeValidations(uid, requestedMethod):
    print("uid: " + uid)
    if (ValidateUser(uid)):
        if(ValidateUserPlan(uid, requestedMethod)):
            return True
        
    return False


def ValidateUser(uid):
    user_ref = db.collection(u'users').document(u'' + uid)
    
    if user_ref.get().to_dict() is None:
        print("user not validated")
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
        print("oldpost:" + oldPost["username"])
        try:
            if oldPost["endDate"] > datetime.now():
                if oldPost[requestedMethod] > 0:
                    oldPost[requestedMethod] -= 1
                    db.collection(u'users').document(u'' + uid).set(dict(oldPost))
                    return True
                else:
                    print("olmadı")
                    return False
            else:
                return False
        except Exception as e:
            return False
    else:
        return True
        
    
        
def LoadScalerFrom(scalerPath):
    s3 = boto3.resource("s3").Bucket("churn-bucket")
    scaler = pickle.loads(s3.Object(key=scalerPath).get()["Body"].read())
    
    return scaler


def LoadModelFrom(modelPath):
    s3 = boto3.resource("s3").Bucket("churn-bucket")
    loaded_model = pickle.loads(s3.Object(key=modelPath).get()["Body"].read())
    
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


class GetUserPlan(Resource):
    def post(self):
        data = request.get_json()
        
        uid = data["uid"]
        
        try:
            user_ref = db.collection(u'users').document(u'' + uid)
    
            userdetails = user_ref.get().to_dict()
            userdetails["endDate"] = json.dumps(userdetails["endDate"], indent=4, sort_keys=True, default=str)
            
            return {'info': 1, 'user': userdetails}
            
        except Exception as e:
            return {'info': -1, 'details': str(e)}



class UpdateUserPlan(Resource):
    def post(self):
        data = request.get_json()
        
        uid = data["uid"]
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
            user_ref = db.collection(u'users').document(u'' + uid)
    
            oldPost = user_ref.get().to_dict()
            oldPost["usertype"] = usertype
            oldPost["columnsInfos"] = columnsInfos
            oldPost["train"] = train
            oldPost["predict"] = predict
            oldPost["endDate"] = (datetime.now() + timedelta(days=365))
            
            db.collection(u'users').document(u'' + uid).set(oldPost)
            return {'info': 1}
        except Exception as e:
            return {'info': -1, 'details': str(e)}
    



class ColumnsInfos(Resource):
    def post(self):
        data = request.get_json()
        
        uid = data["uid"]
        columns = data["columns"]
        dataset = data["dataset"]
        
        try:
            if(MakeValidations(uid, 'columnsInfos')):
                print("validation'ı geçtik1")
                data_frame = pd.DataFrame(dataset, columns = columns)
                data_frame = data_frame[columns].apply(pd.to_numeric, errors="ignore")
                
                colInfos = []
                
                print("validation'ı geçtik2")
                for i in range(len(columns)):
                    cat = 0
                    if(data_frame.iloc[:,[i]].values.dtype is np.dtype("object")):
                        cat = 1
                    
                    counterObj = data_frame.iloc[:, i].value_counts()
                    
                    colInfos.append({"name": columns[i], "values": counterObj.keys().tolist(), "counts": counterObj.tolist(), "cat": cat })
                    
                
                print("validation'ı geçtik3")
                return {'info': 1, 'colInfos': colInfos}
            else:
                return {'info': 0}
        except Exception as e:
            print("haaaaa: " + e)
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
                    gms.Run()
                    #run = Thread(target = gms.Run, args = ())
                    #run.start()
                    return {'info': 1}
            else:
                return {'info': 0, 'details': 'Your have reached your limit.'}
        except Exception as e:
            print(str(e))
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
            print("Hata: " + str(e))
            return {'info': -1, 'details': str(e)}
 
    

        
class ModelList(Resource):
    def post(self):
        data = request.get_json()
        
        uid = data["uid"]
        
        try:
            if(MakeValidations(uid, 'modellist')):
                doc = db.collection(u'models').document(u'' + uid)
                
                if doc.get().to_dict() is None:
                    return {"info": 0}
                else:
                    post = doc.get().to_dict()
                    return {"info": 1, "models": post["models"]}
            else:
                return {"info": 0}
        except Exception as e:
            return {'info': -1, 'details': str(e)}        

    


class CheckTrainStatus(Resource):
    def post(self):
        data = request.get_json()

        uid = data["uid"]
        
        try:
            docs = db.collection(u'trainstatus').where(u'capital', u'==', True).get()
            
            if(MakeValidations(uid, 'checkTrainStatus')):
                statuslistCursor = []
                for doc in docs:
                    statuslistCursor.append(doc.to_dict())
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
            

     
        
api.add_resource(MainPage, '/')
api.add_resource(Test, '/test')

api.add_resource(GetUserPlan, '/getUserPlan')
api.add_resource(UpdateUserPlan, '/updateUserPlan')

api.add_resource(ColumnsInfos, '/columnsInfos')
api.add_resource(Train, '/train')
api.add_resource(Predict, '/predict')

api.add_resource(ModelList, '/modelList')
api.add_resource(CheckTrainStatus, '/checkStatus')

