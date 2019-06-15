#!flask/bin/python

"""Alternative version of the RESTful server implemented using the
Flask-RESTful extension."""

from flask import Flask, jsonify, abort, make_response
from flask_restful import Api, Resource, reqparse, fields, marshal
from flask_httpauth import HTTPBasicAuth
import Test
import wget, os
import dynamodb_classifier as ddb

app = Flask(__name__, static_url_path="")
api = Api(app)
auth = HTTPBasicAuth()

@auth.get_password
def get_password(username):
    if username == 'samsaraRD':
        return 'SD'
    return None


@auth.error_handler
def unauthorized():
    # return 403 instead of 401 to prevent browsers from displaying the default
    # auth dialog
    return make_response(jsonify({'message': 'Unauthorized access'}), 403)

tasks  = [
    {
    'id': 1,
        'url': u'https://samsaranc.github.io/images/avatar.jpg',
        'pred': 0
    }
]

task_fields = {
    'url': fields.String,
    'pred': fields.Integer,
    'uri': fields.Url('task'),
    'ddb': fields.Integer
}

class TaskAPI(Resource):
    decorators = [auth.login_required]

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('url', type=str, location='json')
        self.reqparse.add_argument('pred', type=str, location='json')
        super(TaskAPI, self).__init__()

    def post(self):
        args = self.reqparse.parse_args()
        print(args['url'])
        try:
            filename = wget.download(args['url'])
        except Exception as e:
            filename = None
            return make_response(jsonify({'message': filename}), 404)

        if not os.path.exists(filename):
            return make_response(jsonify({'message': 'Wrong URL'}), 404)

        try:
            #prediction = 69420
            prediction = Test.test_url(filename)
        except Exception as e:
            os.remove(filename)
            return make_response(jsonify({'message': 'Classifier'}), 500)

        ddb_resp = ddb.ddb_add(prediction, args['url'], 2)
        try:
            os.remove(filename)
        except:
            return make_response(jsonify({'message': 'Wrong URL'}), 404)

        task = {
            'id': tasks[-1]['id'] + 1,
            'url': args['url'],
            'pred': prediction,
            'ddb': ddb_resp
        }
#        tasks.append(task)

        return {'task': marshal(task, task_fields)}, 201

api.add_resource(TaskAPI, '/todo/api/v1.0/tasks', endpoint='task')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
