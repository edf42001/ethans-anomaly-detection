# Ethan's Anomaly Detection

Ethan's Anomaly Detection is a small toolset that allows developers to detect abnormal changes in time series data. The service that runs this anomaly detection software is sin the cloud, meaning developers only need to download some small client code to get started.


## Getting Started

Ethan's anomaly detection provides an uniform api for anomaly detection. Here is how to use it:

### Overview:

1. Pick a detector type to use by importing it and instantiate it:
```
from ethansanomaly.AnomalyDetectors import LSTMAnomalyDetector

lstm_detector = LSTMAnomalyDetector()
```
2. Train the anomaly detector on nominal data. In this example, the model will be saved to the folder "saved_models/experiment_name/"  
```
data = np.load("training_data.npy")

lstm_detector.train("experiment_name", data)
```
3. Use the saved model to look for anomalies. The model is automatically loaded and used to find anomalies. 
```
data = np.load("testing_data.npy")

anomalies, scores, anomaly_metrics, threshold = lstm_detector.find_anomalies("experiment_name", data)
```
### Data Format:

All data should be in the form of two-dimensional numpy arrays. The first dimension needs to be time, and the second should be the different features of your time series data. So data arrays will have dimensions of [number of points, number of features per point].


## Anomaly Detection Algorithms:

### LSTM Anomaly Detector

The LSTM Anomaly Detector uses a Long-Short Term Memory neural network to predict the next data point. Then it looks at the error between the predictions and the actual values and uses a dynamic thresholding algorithm to detect anomalies. Although the input data may have multiple features, the LSTM Anomaly Detector only makes its predictions on the first feature. This helps pinpoint which feature in the data caused the problem. If you want to make predictions on each feature in the data, train multiple LSTM Anomaly Detectors while swapping out which feature is first.    

### Kernel PCA Anomaly Detector

The Kernel PCA Anomaly Detector uses the fact that there are relationships between different features in the data. By using a high-dimensional kernel, it extracts these relationships in the training data and looks for deviations from these relationships in the testing data. It defines as an anomaly as anything that deviates three sigmas above the average. This detector does not care how the features are ordered, but as such it can not tell you exactly which feature caused the anomaly. 
