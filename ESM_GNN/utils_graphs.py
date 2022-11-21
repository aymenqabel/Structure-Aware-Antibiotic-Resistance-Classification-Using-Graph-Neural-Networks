from Bio import SeqIO
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataset import ARGDataset
import os
import numpy as np
import torch

global feat_dict, AR_NAME_TO_LABEL, IDX_TO_CLASS
feat_dict = {
        "coords":list(range(3)),
        "amino_acid_one_hot": list(range(3, 23)),
        "phi":[23],
        "psi":[24],
        "rsa":[25],
        "asa":[26],
        "b_factor":[27],
        'ss':list(range(28, 36)),
        "hbond_acceptors":[37],
        "hbond_donors":[38],
        'expasy':list(range(39, 39+5)), #60
        # "meiler": list(range(28, 36)),
        
    }
AR_NAME_TO_LABEL = {'MULTIDRUG': 0,
                'AMINOGLYCOSIDE': 1,
                'MACROLIDE': 2,
                'BETA-LACTAM': 3,
                'GLYCOPEPTIDE': 4,
                'TRIMETHOPRIM': 5,
                'FOLATE-SYNTHESIS-INHABITOR': 6,
                'TETRACYCLINE': 7,
                'SULFONAMIDE': 8,
                'FOSFOMYCIN': 9,
                'PHENICOL': 10,
                'QUINOLONE': 11,
                'STREPTOGRAMIN': 12,
                'BACITRACIN': 13,
                'RIFAMYCIN': 14,
                'MACROLIDE/LINCOSAMIDE/STREPTOGRAMIN': 15}
IDX_TO_CLASS = {v: k for k, v in AR_NAME_TO_LABEL.items()}

def idx_to_class(x):
    return IDX_TO_CLASS[x]

def replace(text):
    text = "".join([c if c.isalnum() or c in ["_", ".", "-"] else "_" for c in text])
    text = text.replace('MACROLIDE_LINCOSAMIDE_STREPTOGRAMIN', 'MACROLIDE-LINCOSAMIDE-STREPTOGRAMIN')
    return text

def feature_len(features = ['amino_acid_one_hot']):
    indices_feat = []
    for feat in features:
        indices_feat += feat_dict[feat]
    return len(indices_feat)

def select_features(data, features = ['amino_acid_one_hot'], edges = [1, 1, 1], threshold=7):
    indice = (data.edge_attr[:, 0] < threshold)*(data.edge_attr[:, 1] == edges[0])*edges[0] + (data.edge_attr[:, 2] == edges[1])*edges[1] + \
                (data.edge_attr[:, 3] == edges[2])*edges[2]
    indice = indice.cpu().numpy()
    indices = np.where(indice >= 1)[0]
    edge_index = data.edge_index[:,indices]
    new_edges = torch.abs(edge_index[0] - edge_index[1]).unsqueeze(1)
    edge_attr = data.edge_attr[indices]
    edge_attr = torch.cat((edge_attr, new_edges), dim = 1)
    indices_feat = []
    for feat in features:
        indices_feat += feat_dict[feat]
    x = data.x[:, indices_feat]
    return x, edge_index, edge_attr

def load_graphs_split(input_file_fasta, names):
    fasta_sequences= list(SeqIO.parse(open(input_file_fasta),'fasta'))
    X = []
    for protein in tqdm(fasta_sequences):
        des = protein.description
        input_file = "_".join(replace(des).split("_")[:4])
        element = names[input_file]
        X.append(element)
    return X

def load_graphs_COALA(path_data = '../../data/splits1', path_PDB = "../../../PDBFiles/", features = ['phi', 'psi','rsa', 'asa', 'b_factor', 'ss','hbond_acceptors', 'hbond_donors', 'expasy'],
seed = 0):
    dataset= ARGDataset(path_PDB, features = features)
    names = {}
    for i in range(len(dataset)):
        names["_".join(dataset[i].name[0].split("_")[:4])] = dataset[i]
    input_file_train = os.path.join(path_data, f'train_file_{seed}.fasta')
    input_file_test = os.path.join(path_data, f'test_file_{seed}.fasta')
    X = load_graphs_split(input_file_train, names)
    X_test = load_graphs_split(input_file_test, names)
    X_train, X_val = train_test_split(X,
        test_size=0.1, shuffle=True, random_state = 0)
    return X_train, X_val, X_test
