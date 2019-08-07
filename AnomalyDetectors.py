from ethansanomaly.AnomalyDetectorClient import AnomalyDetectorClient


class LSTMAnomalyDetector(AnomalyDetectorClient):
    def __init__(self, sequence_len=250, p_threshold=0.13):
        # Params:
        # sequence_len: Number of past data points to use as input to predict the next point
        # p_threshold: Adjusts how many anomalies are pruned. Good values are between (0.05, 0.15)
        super().__init__("LSTMAnomalyDetector", sequence_len=sequence_len, p_threshold=p_threshold)


class KernelPCAAnomalyDetector(AnomalyDetectorClient):
    def __init__(self, window_size=300, stride=40):
        # Params:
        # window_size: How many points in the window the kernel feature extraction is performed on
        # stride: How many points to shift the window when looking for anomalies
        super().__init__("KernelPCAAnomalyDetector", window_size=window_size, stride=stride)
