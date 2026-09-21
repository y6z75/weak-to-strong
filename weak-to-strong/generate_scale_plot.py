from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out_dir = Path(__file__).resolve().parent

# Data Scale Comparison: N=1,000 vs N=5,000
splits = ["Restricted Regime\n(N = 1,000)", "Expanded Regime\n(N = 5,000)"]
student_acc = [59.0, 60.5]      # Student performance under Pythia-70M supervision
pgr_scores = [50.8, 63.1]       # Recovered capability percentage

x = np.arange(len(splits))
w = 0.35

fig, ax1 = plt.subplots(figsize=(7.5, 4.8))

# Accuracy Bars
rects1 = ax1.bar(x - w/2, student_acc, w, label="Student Accuracy (%)", color="#1f77b4", edgecolor="black", alpha=0.85)
ax1.set_ylabel("Student Accuracy (%)", color="#1f77b4", fontsize=10, fontweight="bold")
ax1.set_ylim(40, 75)
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.set_xticks(x)
ax1.set_xticklabels(splits, fontsize=10, fontweight="bold")

# PGR Secondary Axis
ax2 = ax1.twinx()
rects2 = ax2.bar(x + w/2, pgr_scores, w, label="PGR (%)", color="#2ca02c", edgecolor="black", alpha=0.85)
ax2.set_ylabel("Performance Gap Recovered (PGR %)", color="#2ca02c", fontsize=10, fontweight="bold")
ax2.set_ylim(0, 80)
ax2.tick_params(axis="y", labelcolor="#2ca02c")

# Value annotations
for rect in rects1:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2, h + 0.8, f"{h:.1f}%", ha="center", fontweight="bold", fontsize=9)

for rect in rects2:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2, h + 0.8, f"{h:.1f}%", ha="center", fontweight="bold", fontsize=9)

# Baseline line
ax1.axhline(52.8, color="red", linestyle="--", linewidth=1.2, label="Pythia-70M Baseline (52.8%)")

# Joint legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.title("Impact of Training Data Scale on W2SG (Pythia-70M -> GPT-2 Base)", fontweight="bold", fontsize=11)
fig.tight_layout()

scale_path = out_dir / "data_scale_comparison.png"
fig.savefig(str(scale_path), dpi=200)
plt.close(fig)
print(f"Saved: {scale_path.name}")