import torch
from torch_geometric.data import Dataset, InMemoryDataset, Data
from tqdm import tqdm
import json
import numpy as np
import multiprocessing
import os

def _mp_graph_constructor(args):
    return torch.load(args)

class ARGDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, features=[], edges=[]):
        with open(os.path.join(root, "list_elements_coala.json"), "r") as fp:
            self.structures = json.load(fp)
        self.edges = edges
        self.features = features
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) :
        """Names of raw files in the dataset."""
        return [f"data_{pdb}t15.pt" for pdb in self.structures]

    @property
    def processed_file_names(self):
        """Names of processed files to look for"""
        feat_name = "_".join(self.features)
        edge_name = "_".join(self.edges)
        return [f'data_{feat_name}_{edge_name}.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        pool = multiprocessing.Pool(16)
        graphs = []
        y = []
        for result in tqdm(pool.imap_unordered(_mp_graph_constructor, self.raw_paths), total=len(self.raw_paths)):
            g = Data(edge_index = result.edge_index, num_nodes = len(result.node_id), node_id = result.node_id,
            edge_attr = torch.cat((torch.FloatTensor(result.distance).reshape(-1,1), torch.tensor(result.ohe_kind)), dim=1), y = result.label)
            x = torch.cat((torch.FloatTensor(result.amino_acid_one_hot), torch.FloatTensor(result.coords[0])), dim=1)
            for feat in self.features:
                feature = np.array(result[feat])
                if len(feature.shape)==1:
                    feature = feature.reshape(-1,1)
                x = torch.cat((x, torch.FloatTensor(feature)), dim = 1)
            g.x = x

            graphs.append(g)

        pool.close()
        pool.join()

        print("Saving Data...")
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])
        print("Done!")

