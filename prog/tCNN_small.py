import os, csv
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr

print("STARTING SCRIPT...")

# ===== SMALL SETTINGS =====
MAX_SAMPLES = 800
MAX_GENES = 300
FP_BITS = 64

DPATH = '../data' if not os.path.exists('./data') else './data'

IC50_file = f'{DPATH}/CCLE/GDSC_IC50.csv'
Mutation_file = f'{DPATH}/CCLE/genomic_mutation_34673_demap_features.csv'
Smiles_file = f'{DPATH}/223drugs_pubchem_smiles.txt'
Drug_info_file = f'{DPATH}/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'


# ============================================================
# Load SMILES (pubchem -> smiles)
# ============================================================
def load_smiles():
    smi = {}
    for line in open(Smiles_file):
        parts = line.strip().split()
        if len(parts) >= 2:
            smi[parts[0]] = parts[1]
    print("Loaded SMILES:", len(smi))
    return smi


# ============================================================
# DrugID -> PubChem mapping (IMPORTANT FIX)
# ============================================================
def load_drug_mapping():
    mapping = {}
    with open(Drug_info_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 5 and row[5].isdigit():
                drug_id = row[0]
                pubchem = row[5]
                mapping[drug_id] = pubchem
    print("Loaded drug mapping:", len(mapping))
    return mapping


# ============================================================
# Fingerprint (no RDKit)
# ============================================================
def fingerprint(smiles):
    arr = np.zeros(FP_BITS)
    for i, ch in enumerate(smiles):
        arr[(ord(ch)*(i+1)) % FP_BITS] = 1
    return arr


# ============================================================
# Build dataset (CORRECT)
# ============================================================
def build_data():
    print("Loading data...")

    mutation = pd.read_csv(Mutation_file, index_col=0)
    mutation = mutation.iloc[:, :MAX_GENES]

    ic50 = pd.read_csv(IC50_file, index_col=0)

    smiles_map = load_smiles()
    drug_map = load_drug_mapping()

    print("Total drugs:", len(ic50.index))
    print("Total cells:", len(ic50.columns))

    X, y = [], []

    for drug in ic50.index:
        try:
            drug_id = drug.split(':')[1]
        except:
            continue

        if drug_id not in drug_map:
            continue

        pubchem = drug_map[drug_id]

        if pubchem not in smiles_map:
            continue

        smi = smiles_map[pubchem]
        fp = fingerprint(smi)

        for cell in ic50.columns:
            if cell not in mutation.index:
                continue

            val = ic50.loc[drug, cell]
            if np.isnan(val):
                continue

            mut = mutation.loc[cell].values

            X.append(np.concatenate([mut, fp]))
            y.append(val)

            if len(X) % 100 == 0:
                print("Collected:", len(X))

            if len(X) >= MAX_SAMPLES:
                break

        if len(X) >= MAX_SAMPLES:
            break

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print("Final data shape:", X.shape)

    return X, y


# ============================================================
# Split
# ============================================================
def split_data(X, y):
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


# ============================================================
# Model
# ============================================================
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ============================================================
# MAIN
# ============================================================
def run():
    X, y = build_data()

    if len(X) == 0:
        print("ERROR: No data generated → mapping issue")
        return

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Train:", len(X_train), "Test:", len(X_test))

    model = build_model(X.shape[1])

    print("Training...")
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=1)

    print("Testing...")
    y_pred = model.predict(X_test).flatten()

    pcc = pearsonr(y_pred, y_test)[0]
    mse = np.mean((y_pred - y_test) ** 2)

    print("\nFINAL RESULT")
    print("Pearson:", round(pcc, 4))
    print("MSE:", round(mse, 4))


if __name__ == "__main__":
    run()