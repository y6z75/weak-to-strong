from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out_dir = Path(__file__).resolve().parent

# -------------------------------------------------------------
# Plot 1: Loss Function Ablation (xent vs product vs logconf)
# -------------------------------------------------------------
objectives = ["Standard Cross-Entropy\n(xent)", "Product Agreement\n(product)", "Confidence Regularized\n(logconf)"]
accuracies = [60.4, 55.0, 53.5]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

fig1, ax1 = plt.subplots(figsize=(7.5, 4.8))
bars = ax1.bar(objectives, accuracies, color=colors, width=0.55, edgecolor="black", alpha=0.85)

# Add baseline references
ax1.axhline(y=52.8, color="gray", linestyle="--", linewidth=1.2, label="Pythia-70M Baseline (52.8%)")
ax1.axhline(y=65.0, color="darkgreen", linestyle=":", linewidth=1.2, label="GPT-2 GT Ceiling (65.0%)")

ax1.set_ylabel("Student Accuracy (%)", fontsize=10)
ax1.set_ylim(45, 70)
ax1.set_title("W2SG Performance by Loss Function Objective (SciQ)", fontweight="bold", fontsize=11)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.legend(loc="upper right")

for bar in bars:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.6, f"{h:.1f}%", ha="center", fontweight="bold", fontsize=10)

fig1.tight_layout()
loss_obj_path = out_dir / "loss_ablation_comparison.png"
fig1.savefig(str(loss_obj_path), dpi=200)
plt.close(fig1)
print(f"Saved: {loss_obj_path.name}")

# -------------------------------------------------------------
# Plot 2: PGR Divergence (Systematic LLM vs. Random Noise)
# -------------------------------------------------------------
conditions = [
    "Pythia-70M\n(Systematic)",
    "10% Random\nNoise",
    "20% Random\nNoise",
    "30% Random\nNoise"
]
pgr_values = [50.8, -18.2, -190.0, -105.0]
bar_colors = ["#2ca02c" if v > 0 else "#d62728" for v in pgr_values]

fig2, ax2 = plt.subplots(figsize=(8, 4.8))
bars2 = ax2.bar(conditions, pgr_values, color=bar_colors, width=0.5, edgecolor="black", alpha=0.85)

ax2.axhline(0, color="black", linewidth=1)
ax2.set_ylabel("Performance Gap Recovered (PGR %)", fontsize=10)
ax2.set_title("Performance Gap Recovered: Systematic vs. Synthetic Label Noise", fontweight="bold", fontsize=11)
ax2.grid(axis="y", linestyle="--", alpha=0.4)

for bar, val in zip(bars2, pgr_values):
    offset = 6 if val > 0 else -16
    ax2.text(bar.get_x() + bar.get_width() / 2, val + offset, f"{val:.1f}%", ha="center", fontweight="bold", fontsize=9)

ax2.set_ylim(-220, 80)
fig2.tight_layout()
pgr_path = out_dir / "pgr_divergence_comparison.png"
fig2.savefig(str(pgr_path), dpi=200)
plt.close(fig2)
print(f"Saved: {pgr_path.name}")