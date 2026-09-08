import pandas as pd

df = pd.read_csv('experiment_summary.csv')

# Filter ground truth baselines
gt = df[df['Supervision'] == 'ground_truth'].set_index('Model')['Test Accuracy'].to_dict()

print('\n=== GROUND TRUTH CEILINGS ===')
for m, acc in gt.items():
    print(f'{m}: {acc:.4f}')

# Filter weak-to-strong student runs
w2s = df[df['Supervision'] != 'ground_truth']

results = []
for _, row in w2s.iterrows():
    weak_m = row['Supervision']
    strong_m = row['Model']
    student_acc = row['Test Accuracy']
    
    weak_acc = gt.get(weak_m)
    strong_ceiling = gt.get(strong_m)
    
    if weak_acc is not None and strong_ceiling is not None and strong_ceiling != weak_acc:
        pgr = (student_acc - weak_acc) / (strong_ceiling - weak_acc)
    else:
        pgr = None
        
    results.append({
        'Weak Supervisor': weak_m,
        'Strong Student': strong_m,
        'Weak Acc': weak_acc,
        'Student Acc': student_acc,
        'Strong Ceiling': strong_ceiling,
        'PGR': round(pgr, 4) if pgr is not None else 'N/A'
    })

res_df = pd.DataFrame(results)
print('\n=== WEAK-TO-STRONG GENERALIZATION (PGR) ===')
print(res_df.to_string(index=False))
res_df.to_csv('pgr_summary.csv', index=False)
print('\nSaved PGR table to pgr_summary.csv')
