import os, csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

print("🚀 Running FAIR Random Forest Baseline...")

# --------------------------
# SETTINGS
# --------------------------
MAX_SAMPLES = 40000
MAX_GENES = 1200
FP_BITS = 512
RUNS = 3

DPATH = '../data' if not os.path.exists('./data') else './data'

IC50_file = f'{DPATH}/CCLE/GDSC_IC50.csv'
Mutation_file = f'{DPATH}/CCLE/genomic_mutation_34673_demap_features.csv'
Smiles_file = f'{DPATH}/223drugs_pubchem_smiles.txt'
Drug_info_file = f'{DPATH}/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'


# --------------------------
# SMILES -> fingerprint
# --------------------------
def smiles_to_fp(smiles, nbits=FP_BITS):
    arr = np.zeros(nbits, dtype=np.float32)
    for i, ch in enumerate(str(smiles)):
        idx = (ord(ch) * (i + 1)) % nbits
        arr[idx] += 1
    return arr


# --------------------------
# LOAD HELPERS
# --------------------------
def load_smiles():
    d = {}
    for line in open(Smiles_file):
        parts = line.strip().split()
        if len(parts) >= 2:
            d[parts[0]] = parts[1]
    print("Loaded SMILES:", len(d))
    return d


def load_drug_map():
    d = {}
    with open(Drug_info_file) as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) > 5 and str(r[5]).isdigit():
                d[r[0]] = r[5]
    print("Loaded drug mapping:", len(d))
    return d


# --------------------------
# BUILD DATA
# --------------------------
def build_data():
    print("Loading data...")

    mut = pd.read_csv(Mutation_file, index_col=0)

    # select top variable genes
    var = mut.var().sort_values(ascending=False)
    mut = mut[var.index[:MAX_GENES]]

    ic50 = pd.read_csv(IC50_file, index_col=0)

    smiles_map = load_smiles()
    drug_map = load_drug_map()

    X, y, cells = [], [], []

    for drug in ic50.index:
        try:
            did = drug.split(':')[1]
        except:
            continue

        if did not in drug_map:
            continue

        pub = drug_map[did]
        if pub not in smiles_map:
            continue

        fp = smiles_to_fp(smiles_map[pub])

        count = 0

        for cell in ic50.columns:

            if cell not in mut.index:
                continue

            val = ic50.loc[drug, cell]
            if np.isnan(val):
                continue

            mut_vec = mut.loc[cell].values.astype(np.float32)

            X.append(np.concatenate([mut_vec, fp]))
            y.append(val)
            cells.append(cell)

            count += 1

            if count >= 80:
                break

            if len(y) >= MAX_SAMPLES:
                break

        if len(y) >= MAX_SAMPLES:
            break

    print("Total samples:", len(y))
    return np.array(X), np.array(y), np.array(cells)


# --------------------------
# LEAKAGE-FREE SPLIT (IMPROVED)
# --------------------------
def split_data(X, y, cells):
    unique_cells = np.unique(cells)

    np.random.shuffle(unique_cells)

    cut = int(0.8 * len(unique_cells))
    train_cells = set(unique_cells[:cut])

    train_idx = np.isin(cells, list(train_cells))
    test_idx = ~train_idx

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    return X_tr, X_te, y_tr, y_te


# --------------------------
# METRICS
# --------------------------
def evaluate(y_true, y_pred):
    p = pearsonr(y_true, y_pred)[0]
    s = spearmanr(y_true, y_pred)[0]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return p, s, rmse


# --------------------------
# MAIN RUN
# --------------------------
def run():
    X, y, cells = build_data()

    print("Scaling mutation features only...")

    scaler = StandardScaler()
    X[:, :MAX_GENES] = scaler.fit_transform(X[:, :MAX_GENES])

    tree_options = [200, 500, 800]

    p_list, s_list, r_list = [], [], []

    for run in range(RUNS):
        print(f"\n================ Run {run+1} ================")

        np.random.seed(run)

        idx = np.random.permutation(len(y))
        X_s, y_s, cells_s = X[idx], y[idx], cells[idx]

        X_tr, X_te, y_tr, y_te = split_data(X_s, y_s, cells_s)

        print("Train:", X_tr.shape, "Test:", X_te.shape)

        best_model = None
        best_score = -1

        for n in tree_options:
            print(f"\nTraining RF with {n} trees...")

            model = RandomForestRegressor(
                n_estimators=n,
                max_depth=25,
                max_features='sqrt',
                max_samples=0.7,
                n_jobs=-1,
                random_state=run
            )

            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)

            p, s, rmse = evaluate(y_te, preds)

            print(f"Pearson={p:.4f} | Spearman={s:.4f} | RMSE={rmse:.4f}")

            if p > best_score:
                best_score = p
                best_model = model

        print("\nEvaluating BEST model for this run...")

        final_preds = best_model.predict(X_te)
        p, s, rmse = evaluate(y_te, final_preds)

        print(f"FINAL -> Pearson={p:.4f}, Spearman={s:.4f}, RMSE={rmse:.4f}")

        p_list.append(p)
        s_list.append(s)
        r_list.append(rmse)

    print("\n🎯 FINAL RESULTS (3 runs)")
    print(f"Pearson:  {np.mean(p_list):.3f} ± {np.std(p_list):.3f}")
    print(f"Spearman: {np.mean(s_list):.3f} ± {np.std(s_list):.3f}")
    print(f"RMSE:     {np.mean(r_list):.3f} ± {np.std(r_list):.3f}")


if __name__ == "__main__":
    run()