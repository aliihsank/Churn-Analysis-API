import os
from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api

app = Flask(__name__)
app.debug = False
MONGO_URL = os.environ.get('MONGO_URL')
api = Api(app)
CORS(app)

import churnapi.resources
    
