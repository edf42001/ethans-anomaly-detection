import numpy as np
import glob
import os.path
import matplotlib.pyplot as plt

from ethansanomaly.AnomalyDetectors import LSTMAnomalyDetector

###############################################
#
# Anomaly detection on satellite telemetry data
# using Ethan's Anomaly Detection
#
# Author: Ethan Frank
#
# Download the data here:
# https://s3-us-west-2.amazonaws.com/telemanom/data.zip
#
# Data provided by:
# https://github.com/khundman/telemanom
#
###############################################


# Training all 82 models would take upwards of 8 hours. Leave channels empty to train or test all models.
# To train or test only certain models, put the names of the channels. Ex: channels = ["A-1", "A-2", "E-11", "P-14"]
channels = []

# Set to true to train models for the channels in channels. If false, look for anomalies in the channel test data
train = False

sequence_len = 250  # Number of past points to use as input for predicting the next point
p = 0.13  # How much to prune anomalies. Reasonable values are between 0.05 and 0.15

if train:
    for path in glob.glob(os.path.join("data", "train", "*")):
        name = os.path.splitext(os.path.basename(path))[0]  # Get the channel id

        data = np.load(path)  # Load the data

        if channels and name not in channels:  # If channels has names, skip everything but those names
            continue

        # Uncomment to plot the channel's telemetry data
        # plt.plot(data[:, 0])
        # plt.title(name)
        # plt.show()

        detector = LSTMAnomalyDetector(sequence_len=sequence_len, p_threshold=p)  # Define lstm anomaly detector

        detector.train(name, data)  # Train the detector on the data with the channel id as the experiment name
else:
    for path in glob.glob(os.path.join("data", "test", "*")):
        name = os.path.splitext(os.path.basename(path))[0]  # Get the channel id

        data = np.load(path)  # Load the data

        if channels and name not in channels:  # If channels has names, skip everything but those names
            continue

        detector = LSTMAnomalyDetector(sequence_len=sequence_len, p_threshold=p)  # Define lstm anomaly detector.

        anomalies, scores, anomaly_metrics, threshold = detector.find_anomalies(name, data)  # Find anomalies

        print("Anomalies for channel " + name + ": " + str(anomalies))

        # Uncomment to plot telemetry data with detected anomalies
        # plt.plot(data[:, 0])
        # plt.vlines(anomalies, -1, 1)
        # plt.show()

        # Uncomment to plot smoothed errors with detected anomalies
        # plt.plot(anomaly_metrics)
        # plt.vlines(anomalies, 0, max(anomaly_metrics) + 0.05)
        # plt.hlines(threshold, 0, len(anomaly_metrics))
        # plt.show()