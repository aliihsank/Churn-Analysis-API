from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import firestore

cred = credentials.Certificate('././secret.json')
default_app = firebase_admin.initialize_app(cred)


db = firestore.client()


app = Flask(__name__)
app.debug = False
api = Api(app)
CORS(app)

import churnapi.resources
