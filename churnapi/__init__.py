from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)
CORS(app)

import churnapi.resources
    
