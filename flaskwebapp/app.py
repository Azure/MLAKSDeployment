
from flask import Flask, request
import logging
import json
import driver

app = Flask(__name__)
predict_for = driver.get_model_api()
 
@app.route("/score", methods = ['POST'])
def scoreRRS():
    """ Endpoint for scoring
    """
    if request.headers['Content-Type'] != 'application/json':
        return Response(json.dumps({}), status= 415, mimetype ='application/json')
    request_input = request.json['input']
    response = predict_for(request_input)
    print(response)
    return json.dumps({'result': response})


@app.route("/")
def healthy():
    return "Healthy"

# LightGBM Version
@app.route('/version', methods = ['GET'])
def version_request():
    return driver.version()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)