from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out_dir = Path(__file__).resolve().parent

# 1. Bar Chart: Capability Hierarchies (Cross-Family vs Intra-Family)
pairs = ["Pythia-70M -> GPT-2", "GPT-2 -> GPT-2 Med"]
weak_acc = [52.8, 60.0]
student_acc = [59.0, 61.2]
ceiling_acc = [65.0, 63.0]
pgr_vals = [50.8, 40.0]

x = np.arange(len(pairs))
w = 0.25

fig, ax1 = plt.subplots(figsize=(8, 4.8))
ax1.bar(x - w, weak_acc, w, label="Weak Supervisor", color="#aec7e8")
ax1.bar(x, student_acc, w, label="Student (W2SG)", color="#1f77b4")
ax1.bar(x + w, ceiling_acc, w, label="Ground Truth Ceiling", color="#2ca02c")

ax1.set_ylabel("Accuracy (%)", fontsize=10)
ax1.set_ylim(40, 75)
ax1.set_xticks(x)
ax1.set_xticklabels(pairs, fontsize=10, fontweight="bold")
ax1.legend(loc="upper left")
ax1.grid(axis="y", linestyle="--", alpha=0.5)

for i in range(len(pairs)):
    ax1.text(x[i], student_acc[i] + 1.2, f"PGR: {pgr_vals[i]}%", ha="center", fontweight="bold", color="#1f77b4")

plt.title("Weak-to-Strong Generalization Across Capability Hierarchies (SciQ)", fontweight="bold", fontsize=11)
plt.tight_layout()

chart_path = out_dir / "gpt2_scale_comparison.png"
fig.savefig(str(chart_path), dpi=200)
plt.close(fig)
print(f"Saved: {chart_path.name}")

# 2. Markdown Table Summary
md_text = """# Cross-Architecture vs Intra-Family Capability Hierarchy

| Model Hierarchy | Weak Supervisor Acc | Student Acc | GT Ceiling | PGR (%) | Transfer Regime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Pythia-70M -> GPT-2 Base | 52.8% | 59.0% | 65.0% | 50.8% | Cross-Architecture (RoPE -> Absolute) |
| GPT-2 Base -> GPT-2 Medium | 60.0% | 61.2% | 63.0% | 40.0% | Intra-Family (Absolute -> Absolute) |
"""

md_path = out_dir / "model_hierarchy_summary.md"
md_path.write_text(md_text, encoding="utf-8")
print(f"Saved: {md_path.name}")