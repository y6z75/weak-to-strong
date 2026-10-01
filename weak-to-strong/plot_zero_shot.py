import matplotlib.pyplot as plt
import numpy as np

# Set consistent publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

categories = [
    "Zero-Shot LM\n(Open-Book / Context)",
    "Zero-Shot LM\n(Closed-Book)",
    "Ground Truth Ceiling\n(Linear Probe, N=1k)",
    "W2SG Student\n(Pythia -> GPT-2)",
    "Weak Supervisor\n(Pythia-70M Baseline)"
]

accuracies = [79.20, 66.60, 65.00, 59.00, 52.80]
colors = ["#2b5c8f", "#4e79a7", "#2ca02c", "#1f77b4", "#aec7e8"]

bars = ax.bar(categories, accuracies, color=colors, width=0.55, edgecolor="black", linewidth=1.2)

# Add random guess baseline line
ax.axhline(50.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="Chance Level (50.0%)")

# Add supervisor baseline reference line
ax.axhline(52.8, color="red", linestyle=":", linewidth=1.4, alpha=0.8, label="Pythia-70M Baseline (52.8%)")

# Value labels above bars
for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 1.0,
        f"{yval:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

ax.set_ylim(40, 90)
ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
ax.set_title("GPT-2 Base Capability Profiling: Generative Zero-Shot vs. Linear Classification Probing (SciQ)", fontsize=13, fontweight="bold", pad=15)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=11)
ax.legend(frameon=True, loc="upper right", fontsize=10)

plt.tight_layout()
output_path = "zero_shot_capability_profile.png"
plt.savefig(output_path, dpi=300)
print(f"Plot saved successfully as: {output_path}")