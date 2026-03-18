"""
Anomaly Detection Module
========================
Uses scikit-learn's Isolation Forest to detect anomalous infrastructure
metrics.  The detector is first trained on synthetically generated *normal*
data, and then used to classify incoming metric samples.
"""

import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Isolation-Forest-based anomaly detector for IT metrics."""

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Parameters
        ----------
        contamination : float
            Expected proportion of outliers in the training set.
        random_state : int
            Seed for reproducibility.
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, n_samples: int = 1000) -> None:
        """
        Train the model on synthetic *normal* infrastructure data.

        Normal ranges (matching the simulator):
            cpu      : 10 – 75
            memory   : 20 – 75
            latency  :  5 – 150
            disk_io  :  5 – 50

        Parameters
        ----------
        n_samples : int
            Number of synthetic normal samples to generate.
        """
        rng = np.random.RandomState(42)

        normal_data = np.column_stack([
            rng.uniform(10, 75, n_samples),    # cpu
            rng.uniform(20, 75, n_samples),    # memory
            rng.uniform(5, 150, n_samples),    # latency
            rng.uniform(5, 50, n_samples),     # disk_io
        ])

        self.model.fit(normal_data)
        print("[INFO] Anomaly detector trained on "
              f"{n_samples} normal samples.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def detect(self, metrics: dict) -> str:
        """
        Classify a single metrics sample.

        Parameters
        ----------
        metrics : dict
            Must contain keys: cpu, memory, latency, disk_io.

        Returns
        -------
        str
            ``"Normal"`` or ``"Anomaly Detected"``
        """
        sample = np.array([[
            metrics["cpu"],
            metrics["memory"],
            metrics["latency"],
            metrics["disk_io"],
        ]])

        prediction = self.model.predict(sample)

        #  1 → inlier (normal),  -1 → outlier (anomaly)
        return "Normal" if prediction[0] == 1 else "Anomaly Detected"


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    detector = AnomalyDetector()
    detector.train()

    # Normal sample
    normal = {"cpu": 45, "memory": 55, "latency": 80, "disk_io": 25}
    print(f"Normal  -> {detector.detect(normal)}")

    # Anomalous sample
    anomaly = {"cpu": 98, "memory": 95, "latency": 450, "disk_io": 90}
    print(f"Anomaly -> {detector.detect(anomaly)}")
