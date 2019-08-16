import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from ethansanomaly.AnomalyDetectors import KernelPCAAnomalyDetector

n_points = 4000

x = np.arange(n_points)
channel_1 = 2 * np.sin(x / 4)  # First channel is sine wave

channel_2 = 1 * np.power(channel_1, 2) + \
            0.5 * np.random.normal(0, 1, size=n_points)  # Second channel is first squared plus some noise

data = np.stack((channel_1, channel_2), axis=-1)  # Combine the data to dimensions [n_points, 2]

# Split into test and train data
train_data = data[:int(n_points/2)]
test_data = data[int(n_points/2):]

test_data[1000:1100, 0] = 0  # Produce an anomaly in first channel of testing data

# Fit a scaler on the training data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_data)

# Use that scaler to scale the train and test data
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# Create kernel pca detector with specified window size and stride
window_size = 100
stride = 20
detector = KernelPCAAnomalyDetector(window_size=window_size, stride=stride)
detector.train('kernel_experiment', train_data)

anomalies, scores, anomaly_metrics, threshold = detector.find_anomalies('kernel_experiment', test_data)

# Print found anomalies and their severity scores
# Anomalies with low scores could be ignored if needed
print(anomalies)
print(scores)

x = [window_size + stride * i for i in range(len(anomaly_metrics))]  # Find x coordinate of each window
plt.plot(x, anomaly_metrics)  # Plot the anomaly metrics
plt.plot(test_data[:, 0])  # Overlay the first channel of the data
plt.vlines(anomalies, -2, 2)  # show anomalies
plt.hlines(threshold, 0, len(test_data))  # Show anomaly threshold
plt.show()
