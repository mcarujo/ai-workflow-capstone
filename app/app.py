import argparse
import os
import re
import json

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

# Machine Learning classes
from data_processing import DataProcessing
from model import ModelTrain, ModelPredict

app = Flask(__name__)


@app.route("/")
@app.route('/index')
def index():
    return render_template('index.html')


# @app.route('/dashboard')
# def dashboard():
#     return render_template('dashboard.html')


@app.route('/train', methods=['GET', 'POST'])
def train():
    """
    Train process starting by a request.
    """
    # Loading the dataset tools
    data_controller = DataProcessing()
    # Creating the ModelTrain class passing the dataset as argument
    training = ModelTrain(data_controller.get_dataframe_to_train())
    # Runing the trainin process
    model, metrics = training.run()
    # Returning results as HTML
    if request.method == 'GET':
        return render_template('train.html', metrics=[('R2', metrics[0]), ('MSE', metrics[1]), ('MAE', metrics[2])])
    else:
        # Returning results as json
        return jsonify({'R2': metrics[0], 'MSE': metrics[1], 'MAE': metrics[2]})


@app.route('/predict/<int:days>', methods=['GET', 'POST'])
def predict(days=10):
    """
    Prediction the next days.
    """
    # Creating the ModelPredict instance
    model = ModelPredict()
    # Predicting in 'days' future
    predictions = model.predict(days)
    # Transforming in a real json object
    predictions_json = json.loads(predictions.to_json(
        orient="records", date_format='iso', double_precision=True))

    if request.method == 'GET':
        # Return the HTML with the predictions
        return render_template('predict.html', predictions=predictions_json, days=days)
    else:
        # Returning results as json
        return jsonify(predictions_json)


if __name__ == '__main__':

    # parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())
    if args["debug"]:
        app.run(debug=True, port=8080)
    else:
        app.run(host='0.0.0.0', threaded=True, port=8080)
