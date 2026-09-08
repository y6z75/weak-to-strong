import os
import pickle
import pandas as pd

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
                    params = dict(part.split("=", 1) for part in folder.split("-") if "=" in part)
                    model = params.get("ms", "unknown")
                    weak_model = params.get("wms", "ground_truth")
                    loss = params.get("l", "xent")
                    records.append({
                        "Model": model,
                        "Supervision": weak_model,
                        "Loss": loss,
                        "Test Accuracy": round(data.get("avg_acc_test", 0.0), 4),
                        "Inference Accuracy": round(data.get("avg_acc_inference", 0.0), 4),
                        "Folder": folder
                    })
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

if records:
    df = pd.DataFrame(records)
    print("\n=== EXPERIMENT SUMMARY ===")
    print(df.to_string(index=False))
    df.to_csv("experiment_summary.csv", index=False)
    print("\nSaved summary to experiment_summary.csv")
else:
    print("No results.pkl files found in", results_dir)
