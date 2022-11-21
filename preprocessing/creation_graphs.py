from functools import partial
import multiprocessing
import os
import os.path as osp
from sklearn.preprocessing import MultiLabelBinarizer

import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm

from graphein.ml.conversion import convert_nx_to_pyg_data
from graphein.protein.config import ProteinGraphConfig, DSSPConfig
from graphein.protein.edges.distance import add_peptide_bonds, add_delaunay_triangulation, add_distance_threshold
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot, meiler_embedding, hydrogen_bond_acceptor, hydrogen_bond_donor
from graphein.protein.features.nodes.dssp import  phi, psi, asa, rsa

def _mp_graph_constructor(args, config):
    func = partial(construct_graph, config=config)
    return func(pdb_path=args), args.split("_")[:-5][-1], "_".join(args.split("/")[-1].split("_")[:5])

class ARGDatasetUnfiltered(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, list_elements_coala=None):
        self.structures = [elt[:-4] for elt in list_elements_coala]
        self.ar_name_to_label = {'MULTIDRUG': 0,
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
                    'MACROLIDE-LINCOSAMIDE-STREPTOGRAMIN': 15}
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self) :
        """Names of raw files in the dataset."""
        return [f"{pdb}.pdb" for pdb in self.structures]

    @property
    def processed_file_names(self):
        """Names of processed files to look for"""
        return [f'data_{"_".join(name.split("_")[:5])}_t15.pt' for name in self.structures]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        constructors = {"deprotonate": False, 
            "node_metadata_functions": [amino_acid_one_hot, hydrogen_bond_acceptor, hydrogen_bond_donor, meiler_embedding],
            "edge_construction_functions": [partial(add_distance_threshold, long_interaction_threshold=0, threshold=15.), add_peptide_bonds, add_delaunay_triangulation],
            "graph_metadata_functions":[phi,psi, asa, rsa],
            "dssp_config": DSSPConfig(),
        }
        keys_to_keep = ['hbond_donors',
                        'rsa',
                        'b_factor',
                        'phi',
                        'edge_index',
                        'node_id',
                        'dist_mat',
                        'hbond_acceptors',
                        'amino_acid_one_hot',
                        'name',
                        'psi',
                        'sequence_A',
                        'asa',
                        'kind',
                        'distance',
                        'coords',
                        'residue_number',
                        'meiler']
        mlb = MultiLabelBinarizer(classes = ['distance_threshold', 'peptide_bond', 'delaunay' ])
        mlb.fit([['delaunay', 'peptide_bond', 'distance_threshold']])
        config = ProteinGraphConfig(**constructors)
        constructor = partial(
                _mp_graph_constructor, config=config
            )
        pool = multiprocessing.Pool(16)
        graphs = []
        for result in tqdm(pool.imap_unordered(constructor, self.raw_paths), total=len(self.raw_paths)):
            g = convert_nx_to_pyg_data(result[0])
            new_g = Data()
            for key in keys_to_keep:
                new_g[key] = g[key]
            new_g['ohe_kind'] = mlb.transform(g.kind)
            new_g.label = self.ar_name_to_label[result[1]]
            torch.save(new_g, osp.join(self.processed_dir, f'data_{result[2]}_t15.pt'))
            graphs.append(new_g)
        pool.close()
        pool.join()
        print("Done!")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        name = "_".join(self.structures[idx].split('_')[:5])
        data = torch.load(osp.join(self.processed_dir, f'data_{name}_t15.pt'))
        return data

if __name__ == "__main__":
    list_elements_coala = os.listdir('raw')
    dataset= ARGDataset('',list_elements_coala=list_elements_coala)
    