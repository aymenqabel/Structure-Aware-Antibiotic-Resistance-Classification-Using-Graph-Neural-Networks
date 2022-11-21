import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,  confusion_matrix
from Bio import SeqIO
from tqdm import tqdm
import json

def replace(text):
    text = "".join([c if c.isalnum() or c in ["_", ".", "-"] else "_" for c in text])
    text = text.replace('MACROLIDE_LINCOSAMIDE_STREPTOGRAMIN', 'MACROLIDE-LINCOSAMIDE-STREPTOGRAMIN')
    return text

def load_from_fasta(input_file):
    fasta_sequences= list(SeqIO.parse(open(input_file),'fasta'))
    protein_to_class = {}
    real_name_to_name = {}
    for protein in tqdm(fasta_sequences):
        name, classe = protein.id, protein.description.split("|")[-1]
        protein_to_class[name] = classe   
        real_name_to_name[name] = "_".join(replace(protein.description).split("_")[:4])
    return protein_to_class, real_name_to_name

def load_sequences_from_fasta(input_file):
    fasta_sequences= list(SeqIO.parse(open(input_file),'fasta'))
    X, y, names = [], [], []
    for protein in tqdm(fasta_sequences):
        sequence, classe, des = str(protein.seq), protein.description.split("|")[-1], protein.description
        names.append("_".join(replace(des).split("_")[:4]))
        X.append(sequence)
        y.append(classe)
    return X, y, names

def load_COALA(path_folder ,seed = 0):
    input_file_train = os.path.join(path_folder , f'train_file_{seed}.fasta')
    input_file_test = os.path.join(path_folder , f'test_file_{seed}.fasta')

    train_dict, _ = load_from_fasta(input_file_train)   
    test_dict, names_test = load_from_fasta(input_file_test)    

    return train_dict, test_dict, names_test

def load_sequences_COALA(path_folder , val_size = 0.1, seed = 0):
    input_file_train = os.path.join(path_folder , f'train_file_{seed}.fasta')
    input_file_test = os.path.join(path_folder , f'test_file_{seed}.fasta')
    
    x_train, Y_train, _ = load_sequences_from_fasta(input_file_train)
    X_test, y_test, names_test = load_sequences_from_fasta(input_file_test)
    if val_size == 0:
        return x_train,  X_test, Y_train,  y_test, names_test
    X_train, X_val, y_train, y_val = train_test_split(x_train, Y_train, test_size=val_size, random_state=seed, stratify = Y_train)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, names_test

def save_to_json(results, path_results, method, seed):
    file_path = os.path.join(path_results, f"{method}/results_{method}_{seed}.json")
    with open(file_path, 'w') as f:
        json.dump(results, f)

def open_json(input_file):
    with open(input_file) as f:
        dictionary = json.load(f)
    return dictionary

def get_results_identity(results, y_true, y_pred, names, seed, evalue):
    identities = []
    identity_dict = open_json(f'../../data/splits1/test_identity_dict_{seed}_{evalue}.json')
    for name in names:
        identities.append(identity_dict[name])
    identities = np.array(identities)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    indices_less_than_50 = np.where((identities>= 0) & (identities <50))[0]
    results[f'accuracy_less_than_50_{evalue}'] = accuracy_score(y_true[indices_less_than_50], y_pred[indices_less_than_50])
    indices_greater_than_50 = np.where(identities>= 50)[0]
    results[f'accuracy_greater_than_50_{evalue}'] = accuracy_score(y_true[indices_greater_than_50], y_pred[indices_greater_than_50])
    indices_not_found = np.where(identities == -1)[0]
    results[f'accuracy_not_found_{evalue}'] = accuracy_score(y_true[indices_not_found], y_pred[indices_not_found])
    return results, identities

def compute_metrics(y_true, y_pred, path_results, seed, method = "BLAST", names=None, name_result=0):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm,
                        index = np.unique(y_true + y_pred), 
                        columns = np.unique(y_true + y_pred))
    # Plot the confusion Matrix
    plt.figure(figsize=(10,6))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.savefig(os.path.join(path_results, f"{method}/confusion_matrix_{name_result}.png"))
    # Compute metrics and save then in a json file:
    results = {}
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred)
    results['precision'], results['recall'], results['fscore'] = precision.tolist(), recall.tolist(), fscore.tolist()
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
    results['F1macro'] = np.mean(results['fscore'])
    print(f"Balanced accuracy in {seed} set is : {results['balanced_acc']}")
    if names:
        results, identity_3 = get_results_identity(results, y_true, y_pred, names, seed, 1e-3)
        results, identity_7 = get_results_identity(results, y_true, y_pred, names, seed, 1e-7)
        results, identity_10 = get_results_identity(results, y_true, y_pred, names, seed, 1e-10)
        df = pd.DataFrame(list(zip(names, y_true, y_pred, identity_3, identity_7, identity_10)),
               columns =['Names', 'True_class', 'Predicted_class', 'Identity_1e-3', 'Identity_1e-7', 'Identity_1e-10'])
        df.to_csv(os.path.join(path_results, f"{method}/results_{name_result}.csv"))
    # save results to json
    save_to_json(results, path_results, method, name_result)
    return results
