from torch.utils.data import DataLoader, TensorDataset, Dataset

class ESM_Dataset(Dataset):
    def __init__(self, sequences, y, names = None,  transform=None):
        self.sequences = sequences
        self.y = y
        self.names = names

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.names:
            return self.y[idx], self.sequences[idx], self.names[idx]
        return self.y[idx], self.sequences[idx]

global AR_NAME_TO_LABEL, IDX_TO_CLASS

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