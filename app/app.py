import argparse
import json
import logging
import os
import re

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

# Machine Learning classes
from data_processing import DataProcessing
from model import ModelPredict, ModelTrain

app = Flask(__name__)

logging.basicConfig(filename='logs.txt',
                    level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )


@app.route("/")
@app.route('/index')
def index():
    logging.info('Route index.')
    return render_template('index.html')


def flask_logger():
    """
    Creates logging information.
    """


@app.route("/dashboard")
@app.route('/dashboard/<lines>')
def dashboard(lines=100):
    log_file = open("logs.txt", "r")
    file_lines = log_file.readlines()
    file_lines.reverse()
    last_lines = file_lines[:lines]
    log_file.close()
    return render_template('dashboard.html', log_lines=last_lines)


@app.route('/train', methods=['GET', 'POST'])
def train():
    """
    Train process starting by a request.
    """
    logging.info('Route train.')
    # Loading the dataset tools
    logging.info('Loading the dataset tools.')
    data_controller = DataProcessing()
    # Creating the ModelTrain class passing the dataset as argument
    logging.info(
        'Creating the ModelTrain class passing the dataset as argument.')
    training = ModelTrain(data_controller.get_dataframe_to_train())
    # Runing the trainin process
    logging.info('Runing the trainin process.')
    model, metrics = training.run()
    # Returning results as HTML
    logging.info('Returning results as HTML.')
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
    logging.info('Route predict.')
    # Creating the ModelPredict instance
    logging.info('Creating the ModelPredict instance.')
    model = ModelPredict()
    # Predicting in 'days' future
    logging.info(f'Predicting in {days} future.')
    predictions = model.predict(days)
    # Transforming in a real json object
    logging.info('Transforming in a real json object.')
    predictions_json = json.loads(predictions.to_json(
        orient="records", date_format='iso', double_precision=True))
    if request.method == 'GET':
        # Return the HTML with the predictions
        logging.info('Return the HTML with the predictions.')
        return render_template('predict.html', predictions=predictions_json, days=days)
    else:
        # Returning results as json
        logging.info('Returning results as Json.')
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
