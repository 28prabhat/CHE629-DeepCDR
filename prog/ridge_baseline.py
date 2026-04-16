import argparse
import random, os, sys, csv
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Simple Ridge baseline for DeepCDR data
# Usage: python prog/ridge_baseline.py (from project root) or python ridge_baseline.py (from prog/)

TCGA_label_set = ["ALL","BLCA","BRCA","CESC","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG",
                  "LUSC","MESO","MM","NB","OV","PAAD","SCLC","SKCM",
                  "STAD","THCA",'COAD/READ']
# Adjust data path based on current working directory
DPATH = '../data' if not os.path.exists('./data') else './data'
Drug_info_file = '%s/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv' % DPATH
Cell_line_info_file = '%s/CCLE/Cell_lines_annotations_20181226.txt' % DPATH
Cancer_response_exp_file = '%s/CCLE/GDSC_IC50.csv' % DPATH
Genomic_mutation_file = '%s/CCLE/genomic_mutation_34673_demap_features.csv' % DPATH
Smiles_file = '%s/223drugs_pubchem_smiles.txt' % DPATH

def DataSplit(data_idx, ratio=0.95):
    data_train_idx, data_test_idx = [], []
    for each_type in TCGA_label_set:
        data_subtype_idx = [item for item in data_idx if item[-1] == each_type]
        if len(data_subtype_idx) == 0:
            continue
        train_list = random.sample(data_subtype_idx, int(ratio * len(data_subtype_idx)))
        test_list = [item for item in data_subtype_idx if item not in train_list]
        data_train_idx += train_list
        data_test_idx += test_list
    return data_train_idx, data_test_idx

def read_smiles(smiles_path):
    pubchem2smiles = {}
    with open(smiles_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                # format: <pubchem> <smiles>
                pub, smi = parts[0], parts[1]
                pubchem2smiles[pub] = smi
    return pubchem2smiles

def MetadataGenerate(Drug_info_file, Cell_line_info_file, Genomic_mutation_file, Smiles_file):
    # map drug id -> pubchem id (from provided drug list)
    reader = csv.reader(open(Drug_info_file, 'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]: item[5] for item in rows if len(item) > 5 and item[5].isdigit()}

    # map cellline -> cancer type
    cellline2cancertype = {}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        cellline2cancertype[cellline_id] = TCGA_label

    # load mutation features
    mutation_feature = pd.read_csv(Genomic_mutation_file, sep=',', header=0, index_col=[0])

    # load SMILES mapping (pubchem -> smiles)
    pubchem2smiles = read_smiles(Smiles_file)

    # load experiment response table
    experiment_data = pd.read_csv(Cancer_response_exp_file, sep=',', header=0, index_col=[0])
    # filter to drugs that exist in drug list and have SMILES
    drug_match_list = [item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data.loc[drug_match_list]

    data_idx = []
    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            if str(pubchem_id) in pubchem2smiles and each_cellline in mutation_feature.index:
                if not np.isnan(experiment_data_filtered.loc[each_drug, each_cellline]) and each_cellline in cellline2cancertype.keys():
                    ln_IC50 = float(experiment_data_filtered.loc[each_drug, each_cellline])
                    data_idx.append((each_cellline, str(pubchem_id), ln_IC50, cellline2cancertype[each_cellline]))
    print('%d instances across %d cell lines and %d drugs were generated.' % (len(data_idx), len(set([item[0] for item in data_idx])), len(set([item[1] for item in data_idx]))))
    return mutation_feature, pubchem2smiles, data_idx

def fingerprint_from_smiles(smiles, nbits=1024):
    # Try RDKit; otherwise simple deterministic hash-based fingerprint
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        arr = np.zeros((nbits,), dtype=np.int8)
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits)
        onbits = list(bv.GetOnBits())
        arr[onbits] = 1
        return arr
    except Exception:
        # fallback: simple char-hash fingerprint
        arr = np.zeros((nbits,), dtype=np.int8)
        for i, ch in enumerate(smiles):
            idx = (ord(ch) * (i + 1)) % nbits
            arr[idx] = 1
        return arr

def BuildXY(data_idx, mutation_feature, pubchem2smiles, fp_bits=1024):
    X_list = []
    y_list = []
    missing = 0
    for (cell_line_id, pubchem_id, ln_IC50, _) in data_idx:
        smi = pubchem2smiles.get(str(pubchem_id), None)
        if smi is None:
            missing += 1
            continue
        fp = fingerprint_from_smiles(smi, nbits=fp_bits)
        if fp is None:
            missing += 1
            continue
        mut = mutation_feature.loc[cell_line_id].values.astype(np.float32)
        x = np.concatenate([mut, fp.astype(np.float32)])
        X_list.append(x)
        y_list.append(ln_IC50)
    if missing > 0:
        print('Skipped %d samples due to missing/invalid SMILES.' % missing)
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y

def run_ridge():
    random.seed(0)
    mutation_feature, pubchem2smiles, data_idx = MetadataGenerate(Drug_info_file, Cell_line_info_file, Genomic_mutation_file, Smiles_file)
    data_train_idx, data_test_idx = DataSplit(data_idx)

    # Build X/y for train and test
    X_train, y_train = BuildXY(data_train_idx, mutation_feature, pubchem2smiles)
    X_test, y_test = BuildXY(data_test_idx, mutation_feature, pubchem2smiles)

    print('Train samples:', X_train.shape[0], 'Test samples:', X_test.shape[0])
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print('No data for training or testing after filtering. Exiting.')
        return

    # Standardize features (fit on train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    alphas = [0.1, 0.5, 1.0, 5.0]
    results = {}
    for a in alphas:
        model = Ridge(alpha=a)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        pcc = pearsonr(y_pred, y_test)[0]
        mse = np.mean((y_pred - y_test) ** 2)
        print('alpha=%.2f -> Pearson: %.4f, MSE: %.4f' % (a, pcc, mse))
        results[a] = {'pearson': pcc, 'mse': mse}

    # default summary (alpha=1.0)
    if 1.0 in results:
        print('\nChosen alpha=1.0 -> Pearson: %.4f, MSE: %.4f' % (results[1.0]['pearson'], results[1.0]['mse']))

if __name__ == '__main__':
    run_ridge()
