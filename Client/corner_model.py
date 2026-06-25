import numpy as np
from pathlib import Path

DEFAULT_WEIGHTS = Path(__file__).parent / "corner_model_weights.npz"


class CornerModel:
    """MLP minuscule: 9 raycasts -> steering. Pure numpy, zero dependance framework."""

    def __init__(self, weights_path=DEFAULT_WEIGHTS):
        d = np.load(str(weights_path))
        self.W1 = d['W1'].astype(np.float32)
        self.b1 = d['b1'].astype(np.float32)
        self.W2 = d['W2'].astype(np.float32)
        self.b2 = d['b2'].astype(np.float32)
        self.W3 = d['W3'].astype(np.float32)
        self.b3 = d['b3'].astype(np.float32)
        self.ray_max = float(d['ray_max'])

    def predict(self, raycasts):
        x = np.asarray(raycasts, dtype=np.float32) / self.ray_max
        x = np.tanh(x @ self.W1 + self.b1)
        x = np.tanh(x @ self.W2 + self.b2)
        return float(np.tanh(x @ self.W3 + self.b3).ravel()[0])

    @staticmethod
    def load_if_available(path=DEFAULT_WEIGHTS):
        if Path(path).exists():
            return CornerModel(path)
        return None
