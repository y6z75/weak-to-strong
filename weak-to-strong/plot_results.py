import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
os.makedirs("dissertation_plots", exist_ok=True)

# --- PLOT 1: Loss Function Comparison ---
if os.path.exists("experiment_summary.csv"):
    df = pd.read_csv("experiment_summary.csv")
    gpt2_losses = df[(df["Model"] == "gpt2") & (df["Supervision"] == "ground_truth")].copy()

    if not gpt2_losses.empty:
        if "Folder" in gpt2_losses.columns:
            gpt2_losses["Clean_Loss"] = gpt2_losses["Folder"].apply(
                lambda x: "logconf" if "-l=logconf" in str(x) else ("product" if "-l=product" in str(x) else "xent")
            )
        else:
            loss_map = {"lo": "logconf", "pr": "product", "xe": "xent", "5e": "xent"}
            gpt2_losses["Clean_Loss"] = gpt2_losses["Loss"].map(lambda x: loss_map.get(str(x)[:2], str(x)))

        loss_summary = gpt2_losses.groupby("Clean_Loss")["Test Accuracy"].mean().reset_index()

        plt.figure(figsize=(7, 5))
        bars = plt.bar(
            loss_summary["Clean_Loss"],
            loss_summary["Test Accuracy"],
            color=["#4C72B0", "#55A868", "#C44E52"],
            width=0.5,
        )
        plt.title("Experiment 4: Test Accuracy by Loss Function (GPT-2)", pad=15, fontweight="bold")
        plt.xlabel("Loss Function", labelpad=10)
        plt.ylabel("Test Accuracy", labelpad=10)
        plt.ylim(0.40, 0.65)
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 0.005,
                f"{yval:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        plt.tight_layout()
        plt.savefig("dissertation_plots/exp4_loss_comparison.png", dpi=300)
        plt.close()
        print("Saved: dissertation_plots/exp4_loss_comparison.png")

# --- PLOT 2: Weak vs Student vs Ceiling Comparison ---
if os.path.exists("pgr_summary.csv"):
    pgr_df = pd.read_csv("pgr_summary.csv")
    unique_pairs = (
        pgr_df[pgr_df["Weak Supervisor"] != pgr_df["Strong Student"]]
        .drop_duplicates(subset=["Weak Supervisor", "Strong Student"])
        .copy()
    )

    if not unique_pairs.empty:
        plot_data = []
        for _, r in unique_pairs.iterrows():
            weak_name = r["Weak Supervisor"]
            strong_name = r["Strong Student"]
            pair_label = weak_name + " -> " + strong_name

            plot_data.append({"Setup": pair_label, "Type": "Weak Supervisor", "Accuracy": r["Weak Acc"]})
            plot_data.append({"Setup": pair_label, "Type": "Student (W2S)", "Accuracy": r["Student Acc"]})
            plot_data.append({"Setup": pair_label, "Type": "Strong Ceiling", "Accuracy": r["Strong Ceiling"]})

        p_df = pd.DataFrame(plot_data)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=p_df, x="Setup", y="Accuracy", hue="Type", palette="Blues_d")
        plt.title("Weak-to-Strong Supervision Performance Gap", pad=15, fontweight="bold")
        plt.ylabel("Test Accuracy", labelpad=10)
        plt.ylim(0.40, 0.70)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        plt.tight_layout()
        plt.savefig("dissertation_plots/pgr_comparison_bars.png", dpi=300)
        plt.close()
        print("Saved: dissertation_plots/pgr_comparison_bars.png")