"""Entraine le modele de virage sur des donnees manuelles enregistrees.

Usage:
    python train_corner_model.py data/real/run_20240101_120000 [data/real/run_...]

Le script charge les frames + steering depuis les manifests CSV,
calcule les raycasts via la perception, et entraine un petit MLP.
Les poids sont sauvegardes dans Client/corner_model_weights.npz.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Client"))

from live_settings import load_settings
from live_perception import WhiteTapePerception


def load_dataset(run_dirs, perception):
    X, y = [], []
    for run_dir in run_dirs:
        manifest = Path(run_dir) / "manifest.csv"
        if not manifest.exists():
            print("SKIP: manifest introuvable dans {}".format(run_dir))
            continue

        lines = manifest.read_text().splitlines()
        header = lines[0].split(",")
        img_idx = header.index("image_path")
        steer_idx = header.index("steering")

        for line in lines[1:]:
            parts = line.split(",")
            img_path = Path(run_dir) / parts[img_idx]
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            result = perception.process(frame)
            X.append(result.raycast.tolist())
            y.append(float(parts[steer_idx]))

        print("{} : {} exemples charges".format(run_dir, len(lines) - 1))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_numpy_mlp(X, y, hidden1=16, hidden2=8, lr=0.01, epochs=2000):
    """Descente de gradient simple, zero dependance framework."""
    n, n_in = X.shape
    rng = np.random.default_rng(42)

    W1 = rng.normal(0, 0.1, (n_in, hidden1)).astype(np.float32)
    b1 = np.zeros(hidden1, dtype=np.float32)
    W2 = rng.normal(0, 0.1, (hidden1, hidden2)).astype(np.float32)
    b2 = np.zeros(hidden2, dtype=np.float32)
    W3 = rng.normal(0, 0.1, (hidden2, 1)).astype(np.float32)
    b3 = np.zeros(1, dtype=np.float32)

    for epoch in range(epochs):
        # Forward
        h1 = np.tanh(X @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        out = np.tanh(h2 @ W3 + b3).ravel()

        loss = float(np.mean((out - y) ** 2))

        # Backward
        d_out = 2 * (out - y) / n * (1 - out ** 2)
        d_out = d_out[:, None]

        dW3 = h2.T @ d_out
        db3 = d_out.sum(axis=0)
        d_h2 = (d_out @ W3.T) * (1 - h2 ** 2)

        dW2 = h1.T @ d_h2
        db2 = d_h2.sum(axis=0)
        d_h1 = (d_h2 @ W2.T) * (1 - h1 ** 2)

        dW1 = X.T @ d_h1
        db1 = d_h1.sum(axis=0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3

        if epoch % 200 == 0:
            print("epoch {:4d} | loss {:.5f}".format(epoch, loss))

    return W1, b1, W2, b2, W3, b3


def main():
    parser = argparse.ArgumentParser(description="Entraine le modele de virage.")
    parser.add_argument("data_dirs", nargs="+", help="Dossiers de runs enregistres")
    parser.add_argument("--output", default="Client/corner_model_weights.npz")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    settings = load_settings()
    perception = WhiteTapePerception(settings.perception)

    print("=== Chargement des donnees ===")
    X, y = load_dataset(args.data_dirs, perception)
    print("Total: {} exemples".format(len(X)))

    if len(X) == 0:
        print("Aucune donnee trouvee. Verifie les chemins.")
        return

    ray_max = float(X.max()) if X.max() > 0 else 1.0
    X_norm = X / ray_max

    print("\n=== Entrainement ===")
    W1, b1, W2, b2, W3, b3 = train_numpy_mlp(
        X_norm, y, epochs=args.epochs, lr=args.lr
    )

    out_path = ROOT / args.output
    np.savez(str(out_path), W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, ray_max=ray_max)
    print("\nModele sauvegarde: {}".format(out_path))

    # Score rapide
    h1 = np.tanh(X_norm @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    pred = np.tanh(h2 @ W3 + b3).ravel()
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    print("R2 entrainement: {:.3f}".format(r2))


if __name__ == "__main__":
    main()
