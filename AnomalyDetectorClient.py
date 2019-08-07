import requests
import json
import abc
import os.path

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetectorClient:
    def __init__(self, type, **kwargs):
        self.type = type
        self.ip = "data.ethansanomalydetection.com" if True else "localhost"
        self.port = "80"

        self.detector_config = kwargs

    def train(self, experiment_name, data):
        # Trains an anomaly finding model on the training data and saves the model to your local disk
        # Params:
        # experiment_name: The name of the experiment. Used to name the model
        # data: The data to train on
        # **kwargs: Extra parameters
        #
        # Returns: nothing

        logger.info("Making train request for experiment: " + experiment_name)

        payload = {'method': 'train', 'time_series': data.tolist(), 'experiment_name': experiment_name,
                   'detector_type': self.type, "detector_config": self.detector_config}
        r = requests.post("http://{}:{}/api/v0.1/automation/anomaly/core".format(self.ip, self.port), json=payload)

        if r.ok:
            model_serial = json.loads(r.content)  # Extract the serialized model

            self.save_model(experiment_name, model_serial)  # Save model to local disk

            logger.info(self.type + " successfully trained for: " + experiment_name)
        else:
            logger.error("Error while training, response not OK: " + r.reason)

    def find_anomalies(self, experiment_name, data):
        # Looks for anomalies in data
        # Params:
        # experiment_name: The name of the experiment. Used to load the correct model.
        # data: The data to find anomalies in.
        # **kwargs: Extra parameters
        #
        # Returns:
        # anomalies: List of sequences of anomalies
        # scores: List of severity scores for each anomaly
        # anomaly_metrics: List of the anomaly metric for each data point. This is normally something like
        #                  the error between the model's prediction and the actual data for each data point
        # threshold: The threshold used on the anomaly metrics to determine what is and what isn't an anomaly

        model_serial = self.load_model(experiment_name)  # Load model

        logger.info("Making find_anomalies request for experiment: " + experiment_name)

        payload = {'method': 'find_anomalies', 'time_series': data.tolist(),
                   'experiment_name': experiment_name, 'detector_type': self.type,
                   'model_serial': model_serial, "detector_config": self.detector_config}
        r = requests.post("http://{}:{}/api/v0.1/automation/anomaly/core".format(self.ip, self.port), json=payload)

        if r.ok:
            logger.info("Anomalies successfully received using " + self.type + "for experiment: " + experiment_name)
            data = json.loads(r.content)

            # Read Data
            anomalies = data['anomalies']
            scores = data['scores']
            anomaly_metrics = data['anomaly_metrics']
            threshold = data['threshold']

            return anomalies, scores, anomaly_metrics, threshold
        else:
            logger.error("Error while looking for anomalies, response not OK: " + r.reason)
            return [], {}

    def save_model(self, experiment_name, model_serial):

        # Create folder for the experiment if it doesn't exist
        experiment_path = os.path.join("saved_models", experiment_name)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)

        model_path = os.path.join(experiment_path, experiment_name + "_model.txt")

        with open(model_path, 'w') as f:
            f.write(model_serial)

    def load_model(self, experiment_name):
        model_path = os.path.join("saved_models", experiment_name, experiment_name + "_model.txt")

        with open(model_path, 'r') as f:
            model_serial = f.read()

        return model_serial
