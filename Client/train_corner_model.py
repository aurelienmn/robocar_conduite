"""
Entraine le modele de virage depuis les donnees enregistrees par record_driving.py.

Usage:
    python3 train_corner_model.py data_circuit.csv
    python3 train_corner_model.py data1.csv data2.csv data3.csv   # plusieurs circuits

Le modele entraine est sauvegarde dans corner_model_weights.npz.
La voiture l'utilisera automatiquement dans les virages serres.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def load_data(csv_paths):
    X_list, y_list = [], []
    for path in csv_paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            ray_cols = [c for c in reader.fieldnames if c.startswith("ray")]
            ray_cols.sort(key=lambda c: int(c[3:]))
            for row in reader:
                rays = [float(row[c]) for c in ray_cols]
                steering = float(row["steering"])
                X_list.append(rays)
                y_list.append(steering)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def relu(x):
    return np.maximum(0.0, x)


def relu_grad(x):
    return (x > 0).astype(np.float32)


def train(X, y, hidden1=24, hidden2=12, lr=0.002, epochs=800, batch_size=64):
    n_in = X.shape[1]

    rng = np.random.default_rng(42)
    W1 = rng.standard_normal((n_in, hidden1)).astype(np.float32) * np.sqrt(2.0 / n_in)
    b1 = np.zeros(hidden1, dtype=np.float32)
    W2 = rng.standard_normal((hidden1, hidden2)).astype(np.float32) * np.sqrt(2.0 / hidden1)
    b2 = np.zeros(hidden2, dtype=np.float32)
    W3 = rng.standard_normal((hidden2, 1)).astype(np.float32) * np.sqrt(2.0 / hidden2)
    b3 = np.zeros(1, dtype=np.float32)

    n = len(X)
    for epoch in range(epochs):
        idx = rng.permutation(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            Xb = X[batch]
            yb = y[batch].reshape(-1, 1)

            # Forward
            z1 = Xb @ W1 + b1
            a1 = relu(z1)
            z2 = a1 @ W2 + b2
            a2 = relu(z2)
            z3 = a2 @ W3 + b3
            out = np.tanh(z3)

            loss = float(np.mean((out - yb) ** 2))
            total_loss += loss

            # Backward
            d = 2.0 * (out - yb) / len(batch)
            d *= (1.0 - out ** 2)  # tanh grad

            dW3 = a2.T @ d
            db3 = d.sum(axis=0)
            d2 = d @ W3.T * relu_grad(z2)

            dW2 = a1.T @ d2
            db2 = d2.sum(axis=0)
            d1 = d2 @ W2.T * relu_grad(z1)

            dW1 = Xb.T @ d1
            db1 = d1.sum(axis=0)

            W3 -= lr * dW3
            b3 -= lr * db3
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        if (epoch + 1) % 100 == 0:
            batches = max(1, n // batch_size)
            print(f"  epoch {epoch+1:4d}/{epochs}  loss={total_loss/batches:.5f}")

    return W1, b1, W2, b2, W3, b3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("corner_model_weights.npz"))
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.002)
    args = parser.parse_args()

    print(f"Chargement des donnees: {[str(p) for p in args.csv_files]}")
    X, y = load_data(args.csv_files)
    print(f"  {len(X)} frames chargees, {X.shape[1]} rayons")

    ray_max = float(X.max()) if X.max() > 0 else 1.0
    X_norm = X / ray_max

    print(f"  steering: min={y.min():.2f} max={y.max():.2f} mean={y.mean():.2f}")
    print(f"  ray_max={ray_max:.1f}")
    print(f"Entrainement ({args.epochs} epochs)...")

    W1, b1, W2, b2, W3, b3 = train(X_norm, y, epochs=args.epochs, lr=args.lr)

    # Validation rapide
    z1 = np.maximum(0, X_norm @ W1 + b1)
    z2 = np.maximum(0, z1 @ W2 + b2)
    pred = np.tanh(z2 @ W3 + b3).ravel()
    mse = float(np.mean((pred - y) ** 2))
    print(f"MSE final: {mse:.5f}  (RMSE={mse**0.5:.4f})")

    np.savez(str(args.output), W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, ray_max=ray_max)
    print(f"Modele sauvegarde: {args.output}")
    print("La voiture utilisera automatiquement ce modele dans les virages.")


if __name__ == "__main__":
    main()
