import os, csv
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

print("Running final tCNN model...")

# --------------------------
# settings
# --------------------------
MAX_SAMPLES = 25000
MAX_GENES = 1500
MAX_SMILES_LEN = 100
EPOCHS = 10
RUNS = 3
BATCH_SIZE = 32

DPATH = '../data' if not os.path.exists('./data') else './data'

IC50_file = f'{DPATH}/CCLE/GDSC_IC50.csv'
Mutation_file = f'{DPATH}/CCLE/genomic_mutation_34673_demap_features.csv'
Smiles_file = f'{DPATH}/223drugs_pubchem_smiles.txt'
Drug_info_file = f'{DPATH}/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'

# --------------------------
# SMILES encoding
# --------------------------
chars = list("#%()+-./0123456789:=@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz")
char_to_idx = {c:i for i,c in enumerate(chars)}

def encode_smiles(s):
    mat = np.zeros((MAX_SMILES_LEN, len(chars)), dtype=np.float32)
    for i, ch in enumerate(str(s)[:MAX_SMILES_LEN]):
        if ch in char_to_idx:
            mat[i, char_to_idx[ch]] = 1.0
    return mat

# --------------------------
# helpers
# --------------------------
def load_smiles():
    d = {}
    for line in open(Smiles_file):
        parts = line.strip().split()
        if len(parts) >= 2:
            d[parts[0]] = parts[1]
    return d

def load_drug_map():
    d = {}
    with open(Drug_info_file) as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) > 5 and r[5].isdigit():
                d[r[0]] = r[5]
    return d

# --------------------------
# dataset
# --------------------------
def build_data():
    print("Loading data...")

    mut = pd.read_csv(Mutation_file, index_col=0)

    # keep most important genes
    var = mut.var().sort_values(ascending=False)
    mut = mut[var.index[:MAX_GENES]]

    ic50 = pd.read_csv(IC50_file, index_col=0)

    smiles_map = load_smiles()
    drug_map = load_drug_map()

    Xd, Xm, y, cells = [], [], [], []

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

        drug_vec = encode_smiles(smiles_map[pub])

        count = 0
        for cell in ic50.columns:

            if cell not in mut.index:
                continue

            val = ic50.loc[drug, cell]
            if np.isnan(val):
                continue

            Xd.append(drug_vec)
            Xm.append(mut.loc[cell].values.astype(np.float32))
            y.append(val)
            cells.append(cell)

            count += 1

            if count >= 70:
                break

            if len(y) % 3000 == 0:
                print("Collected:", len(y))

            if len(y) >= MAX_SAMPLES:
                break

        if len(y) >= MAX_SAMPLES:
            break

    print("Total samples:", len(y))
    return np.array(Xd), np.array(Xm), np.array(y), np.array(cells)

# --------------------------
# split (no leakage)
# --------------------------
def split_data(Xd, Xm, y, cells):
    uniq = list(set(cells))
    np.random.shuffle(uniq)

    cut = int(0.8 * len(uniq))
    train_cells = set(uniq[:cut])

    Xd_tr, Xm_tr, y_tr = [], [], []
    Xd_te, Xm_te, y_te = [], [], []

    for i in range(len(y)):
        if cells[i] in train_cells:
            Xd_tr.append(Xd[i])
            Xm_tr.append(Xm[i])
            y_tr.append(y[i])
        else:
            Xd_te.append(Xd[i])
            Xm_te.append(Xm[i])
            y_te.append(y[i])

    return map(np.array, (Xd_tr, Xd_te, Xm_tr, Xm_te, y_tr, y_te))

# --------------------------
# model (slightly improved)
# --------------------------
def build_model():
    d_in = tf.keras.Input(shape=(MAX_SMILES_LEN, len(chars)))

    x = tf.keras.layers.Conv1D(64, 5, activation='relu')(d_in)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(128, 5, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(128, 3, activation='relu')(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    m_in = tf.keras.Input(shape=(MAX_GENES,))
    y = tf.keras.layers.Dense(512, activation='relu')(m_in)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Dropout(0.3)(y)
    y = tf.keras.layers.Dense(256, activation='relu')(y)

    c = tf.keras.layers.concatenate([x, y])
    z = tf.keras.layers.Dense(256, activation='relu')(c)
    z = tf.keras.layers.Dropout(0.3)(z)
    z = tf.keras.layers.Dense(128, activation='relu')(z)
    out = tf.keras.layers.Dense(1)(z)

    model = tf.keras.Model(inputs=[d_in, m_in], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

    return model

# --------------------------
# main
# --------------------------
def run():
    Xd, Xm, y, cells = build_data()

    scaler = StandardScaler()
    Xm = scaler.fit_transform(Xm)

    p_list, s_list, r_list = [], [], []

    for seed in range(RUNS):
        print(f"\nRun {seed+1}")

        np.random.seed(seed)
        tf.random.set_seed(seed)

        idx = np.random.permutation(len(y))
        Xd_s, Xm_s, y_s, cells_s = Xd[idx], Xm[idx], y[idx], cells[idx]

        Xd_tr, Xd_te, Xm_tr, Xm_te, y_tr, y_te = split_data(Xd_s, Xm_s, y_s, cells_s)

        print("Training...")
        model = build_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=2,
                restore_best_weights=True
            )
        ]

        model.fit(
            [Xd_tr, Xm_tr], y_tr,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            callbacks=callbacks
        )

        print("Testing...")
        preds = model.predict([Xd_te, Xm_te]).flatten()

        p = pearsonr(preds, y_te)[0]
        s = spearmanr(preds, y_te)[0]
        r = np.sqrt(mean_squared_error(y_te, preds))

        print("Pearson:", round(p,4),
              "Spearman:", round(s,4),
              "RMSE:", round(r,4))

        p_list.append(p)
        s_list.append(s)
        r_list.append(r)

    print("\nFinal Results:")
    print("Pearson:", round(np.mean(p_list),3), "±", round(np.std(p_list),3))
    print("Spearman:", round(np.mean(s_list),3), "±", round(np.std(s_list),3))
    print("RMSE:", round(np.mean(r_list),3), "±", round(np.std(r_list),3))


if __name__ == "__main__":
    run()