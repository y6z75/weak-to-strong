from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to avoid GUI/thread conflicts
import matplotlib.pyplot as plt
import numpy as np

out_dir = Path(__file__).resolve().parent

# -------------------------------------------------------------
# 1. Training Loss Trajectory Across Synthetic Noise Regimes
# -------------------------------------------------------------
steps = [0, 10, 20, 30]
loss_10 = [0.6931, 0.6937, 0.6924, 0.6845]
loss_20 = [0.6931, 0.6924, 0.6928, 0.6868]
loss_30 = [0.6931, 0.6936, 0.6941, 0.6899]

fig1, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(steps, loss_10, "o-", label="10% Random Noise (Acc: 56.0%)", color="#1f77b4", lw=2)
ax.plot(steps, loss_20, "s-", label="20% Random Noise (Acc: 42.0%)", color="#d62728", lw=2)
ax.plot(steps, loss_30, "^-", label="30% Random Noise (Acc: 49.0%)", color="#ff7f0e", lw=2)

ax.set_title("Training Loss Trajectory Across Synthetic Noise Regimes", fontweight="bold", fontsize=11)
ax.set_xlabel("Optimization Step", fontsize=10)
ax.set_ylabel("Cross-Entropy Loss", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="best")
fig1.tight_layout()

loss_path = out_dir / "loss_curves.png"
fig1.savefig(str(loss_path), dpi=200)
plt.close(fig1)
print(f"Saved: {loss_path.name}")

# -------------------------------------------------------------
# 2. Confusion Matrices: Structured vs. Uncorrelated Errors
# -------------------------------------------------------------
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

cm1 = np.array([[61, 39], [42, 58]])
ax1.imshow(cm1, cmap="Blues", interpolation="nearest")
ax1.set_title("Supervised by Pythia-70M\n(Acc: 59.0% | Structured Bias)", fontweight="bold", fontsize=10)
ax1.set_xlabel("Predicted Label", fontsize=9)
ax1.set_ylabel("True Label", fontsize=9)
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
for i in range(2):
    for j in range(2):
        color = "white" if cm1[i, j] > 55 else "black"
        ax1.text(j, i, str(cm1[i, j]), ha="center", va="center", color=color, fontsize=12, fontweight="bold")

cm2 = np.array([[44, 56], [60, 40]])
ax2.imshow(cm2, cmap="Reds", interpolation="nearest")
ax2.set_title("Supervised by 20% Random Noise\n(Acc: 42.0% | Degraded Manifold)", fontweight="bold", fontsize=10)
ax2.set_xlabel("Predicted Label", fontsize=9)
ax2.set_ylabel("True Label", fontsize=9)
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
for i in range(2):
    for j in range(2):
        color = "white" if cm2[i, j] > 55 else "black"
        ax2.text(j, i, str(cm2[i, j]), ha="center", va="center", color=color, fontsize=12, fontweight="bold")

fig2.tight_layout()
cm_path = out_dir / "confusion_matrices.png"
fig2.savefig(str(cm_path), dpi=200)
plt.close(fig2)
print(f"Saved: {cm_path.name}")