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
default_bucket = storage.bucket(name="churn-2537f.appspot.com", app=None)


#doc_ref = db.collection(u'users').document(u'alovelace')
#doc_ref.set({
#    u'first': u'Ada',
#    u'last': u'Lovelace',
#    u'born': 1815
#})
#
#
#users_ref = db.collection(u'users').document(u'asdas')
#print(users_ref.get().to_dict())
#docs = users_ref.get()
#
#for doc in docs:
#    print(u'{} => {}'.format(doc.id, doc.to_dict()))
#
#
#
#blob = default_bucket.blob('my-test-file.txt')
#outfile='C:\\Users\\ali_k\\Desktop\\meraba.txt'
#blob.upload_from_filename(outfile)


app = Flask(__name__)
app.debug = False
api = Api(app)
CORS(app)

import churnapi.resources
