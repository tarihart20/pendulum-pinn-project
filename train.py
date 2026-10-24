#import libraries
import argparse
import re
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

from model.model import PendulumNN

G = 9.81
SEED = 7
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_data(csv_path):
    df = pd.read_csv(csv_path, encoding="latin1")
    df.columns = [str(c).strip() for c in df.columns]

    length_col = next(c for c in df.columns if "string length" in c.lower())
    time_col = next(c for c in df.columns if "period of 10" in c.lower())
    count_col = next(c for c in df.columns if "number of osc" in c.lower())

    df[length_col] = pd.to_numeric(df[length_col], errors="coerce")
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
    measured = df.dropna(subset=[length_col, time_col, count_col]).copy()
    measured = measured[(measured[length_col] > 0) & (measured[count_col] > 0)]

    lengths_m = measured[length_col].to_numpy(dtype=np.float32)
    periods_s = (measured[time_col] / measured[count_col]).to_numpy(dtype=np.float32)
    return lengths_m, periods_s, measured

#main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/data.csv", help="Path to the attached CSV")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--physics-weight", type=float, default=0.02,
                        help="Penalty on correction magnitude; increase to favor theory")
    args = parser.parse_args()

    lengths, periods, used_rows = load_data(args.csv)
    if len(lengths) < 2:
        raise ValueError("Need at least two measured rows with length and timing data.")

    print(f"Loaded {len(lengths)} measured trials.")
    print("Length values are treated as meters based on consistency with the theory.")
    print(f"Lengths (m): {lengths}")
    print(f"Measured periods (s/oscillation): {periods}")

    L = torch.tensor(lengths.reshape(-1, 1), dtype=torch.float32)
    T = torch.tensor(periods.reshape(-1, 1), dtype=torch.float32)

    model = PendulumNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    history = []
    for epoch in range(args.epochs):
        model.train()
        prediction = model(L)
        data_loss = mse(prediction, T)
        correction = model.correction(torch.log(torch.clamp(L, min=1e-6)))
        physics_regularization = torch.mean(correction ** 2)
        loss = data_loss + args.physics_weight * physics_regularization

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append((loss.item(), data_loss.item()))

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1:4d} | total loss={loss.item():.6f} "
                    f"| data MSE={data_loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        fitted = model(L).squeeze(1).numpy()
    rmse = float(np.sqrt(np.mean((fitted - periods) ** 2)))
    print(f"\nTraining RMSE: {rmse:.4f} s per oscillation")
    print("\nLength (m) | measured T (s) | NN fitted T (s) | theory T (s)")
    for length, measured_t, fitted_t in zip(lengths, periods, fitted):
        theory_t = 2 * np.pi * np.sqrt(length / G)
        print(f"{length:10.3f} | {measured_t:14.4f} | {fitted_t:15.4f} | {theory_t:12.4f}")

    torch.save(model.state_dict(), "pendulum_nn.pt")
    print("\nSaved trained weights to pendulum_nn.pt")

    grid = np.linspace(max(0.01, lengths.min() * 0.8), lengths.max() * 1.2, 250)
    grid_t = torch.tensor(grid.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        nn_curve = model(grid_t).squeeze(1).numpy()
    theory_curve = 2 * np.pi * np.sqrt(grid / G)

    plt.figure(figsize=(8, 5))
    plt.scatter(lengths, periods, label="Measured trials", zorder=3)
    plt.plot(grid, nn_curve, label="Physics-informed neural network")
    plt.plot(grid, theory_curve, "--", label="Ideal theory")
    plt.xlabel("Pendulum length (m)")
    plt.ylabel("Period (s per oscillation)")
    plt.title("Simple pendulum: data, neural network, and theory")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pendulum_fit.png", dpi=160)
    print("Saved plot to pendulum_fit.png")
    plt.show()



if __name__ == "__main__":
    main()