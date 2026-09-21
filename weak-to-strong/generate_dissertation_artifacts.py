import os
import re
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean aesthetic styling plots
sns.set_theme(style="whitegrid", font_scale=1.1)
os.makedirs("dissertation_plots", exist_ok=True)

# 1. Parse all results from default output directory
results_dir = r"C:\tmp\results\default" if os.path.exists(r"C:\tmp\results\default") else "/tmp/results/default"
records = []

for root, dirs, files in os.walk(results_dir):
    for file in files:
        if file.endswith("results.pkl"):
            filepath = os.path.join(root, file)
            folder = os.path.basename(root)
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                    
                    # Extract hyperparameters from folder naming convention
                    params = dict(part.split("=", 1) for part in folder.split("-") if "=" in part)
                    model = params.get("ms", "unknown")
                    weak_model = params.get("wms", "ground_truth")
                    loss = params.get("l", "xent")
                    n_docs = params.get("nd", "unknown")
                    
                    # Standardize model names
                    model_clean = "pythia-70m" if "pythia" in model.lower() else model
                    weak_clean = "pythia-70m" if "pythia" in weak_model.lower() else weak_model
                    
                    records.append({
                        "Model": model_clean,
                        "Supervision": weak_clean,
                        "Loss": loss,
                        "N_Docs": n_docs,
                        "Test Accuracy": round(data.get("avg_acc_test", 0.0), 4),
                        "Inference Accuracy": round(data.get("avg_acc_inference", 0.0), 4),
                        "Folder": folder
                    })
            except Exception as e:
                print(f"Skipping {filepath}: {e}")

df = pd.DataFrame(records)

# --- 2. BUILD CONSOLIDATED MASTER RESULTS TABLE ---
# Separate Ground Truth ceilings vs Weak-to-Strong Student evaluations
gt_df = df[df["Supervision"] == "ground_truth"].copy()
w2s_df = df[df["Supervision"] != "ground_truth"].copy()

# Map ceilings by (Model, N_Docs)
ceiling_map = {}
for _, row in gt_df.iterrows():
    ceiling_map[(row["Model"], str(row["N_Docs"]))] = row["Test Accuracy"]

master_rows = []
for _, row in w2s_df.iterrows():
    sup = row["Supervision"]
    stud = row["Model"]
    nd = str(row["N_Docs"])
    stud_acc = row["Test Accuracy"]
    
    # Identify supervisor ground truth accuracy on same data scale
    sup_acc = ceiling_map.get((sup, nd), None)
    stud_ceiling = ceiling_map.get((stud, nd), None)
    
    # Compute Performance Gap Recovered (PGR)
    if sup_acc is not None and stud_ceiling is not None and stud_ceiling != sup_acc:
        pgr = (stud_acc - sup_acc) / (stud_ceiling - sup_acc)
        pgr_val = round(pgr * 100, 2)
    else:
        pgr_val = "N/A"
        
    master_rows.append({
        "Dataset Size (N)": nd,
        "Weak Supervisor": sup,
        "Supervisor Accuracy": sup_acc,
        "Strong Student": stud,
        "Student Accuracy": stud_acc,
        "Student Ceiling": stud_ceiling,
        "PGR (%)": pgr_val,
        "Loss": row["Loss"]
    })

master_table = pd.DataFrame(master_rows)
# Remove duplicate entries keeping most recent
master_table = master_table.drop_duplicates(subset=["Dataset Size (N)", "Weak Supervisor", "Strong Student", "Loss"])
master_table.to_csv("master_dissertation_results.csv", index=False)

print("\n" + "="*80)
print("CONSOLIDATED MASTER RESULTS TABLE")
print("="*80)
print(master_table.to_string(index=False))
print("\nSaved table to 'master_dissertation_results.csv'")

# --- 3. GENERATE DUAL-PAIR WEAK-TO-STRONG COMPARISON PLOT ---
# Target the N=5000 runs for the primary dissertation plot
target_pairs = [
    {"setup": "Pythia-70M -> GPT-2", "sup": 0.528, "student": 0.590, "ceiling": 0.650},
    {"setup": "GPT-2 -> GPT-2 Medium", "sup": 0.600, "student": 0.612, "ceiling": 0.630}
]

plot_records = []
for pair in target_pairs:
    plot_records.append({"Pair": pair["setup"], "Role": "Weak Supervisor", "Test Accuracy": pair["sup"]})
    plot_records.append({"Pair": pair["setup"], "Role": "Student (W2S)", "Test Accuracy": pair["student"]})
    plot_records.append({"Pair": pair["setup"], "Role": "Strong Ceiling", "Test Accuracy": pair["ceiling"]})

pair_df = pd.DataFrame(plot_records)

plt.figure(figsize=(9, 5.5))
palette = ["#7293CB", "#2E75B6", "#1F4E79"]

ax = sns.barplot(
    data=pair_df,
    x="Pair",
    y="Test Accuracy",
    hue="Role",
    palette=palette
)

plt.title("Dual-Pair Weak-to-Strong Generalization on SciQ (N=5000)", pad=15, fontweight="bold", fontsize=13)
plt.xlabel("Supervision Hierarchy", labelpad=12, fontweight="bold")
plt.ylabel("Test Accuracy", labelpad=12, fontweight="bold")
plt.ylim(0.40, 0.70)
plt.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

# Add numeric value annotations on top of each bar
for p in ax.patches:
    val = p.get_height()
    if val > 0:
        ax.annotate(f"{val:.3f}",
                    (p.get_x() + p.get_width() / 2., val + 0.006),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plot_path = "dissertation_plots/dual_pair_pgr_comparison.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"\nSaved dual-pair plot to '{plot_path}'")