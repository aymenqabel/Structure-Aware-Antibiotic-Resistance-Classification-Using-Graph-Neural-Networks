"""
Model and utils function related to the arg_cnn.py file
"""
import numpy as np

import esm
import torch
import torch.nn as nn
import torch.nn.functional as F

class ESM_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        num_features = 2560
        self.model.lm_head = nn.Linear(num_features, 16)
        self.name = False
    
    def transform_to_batch(self, data):
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
        if len(data) == 3:
            names = data[2]
            self.name = True
        new_data = [(data[0][i], data[1][i]) for i in range(len(data[0]))]
        new_batch = self.batch_converter(new_data)
        batch_labels, batch_strs, batch_tokens = new_batch
        y = [AR_NAME_TO_LABEL[elt] for elt in batch_labels]
        if self.name:
            return batch_tokens.long().cuda(), torch.LongTensor(y).cuda(), names
        else:
            return batch_tokens.long().cuda(), torch.LongTensor(y).cuda()

    def forward(self, x):
        if self.name:
            inputs, labels, names = self.transform_to_batch(x)
            out = self.model(inputs)['logits'].mean(1)
            return F.log_softmax(out), labels, names
        else:
            inputs, labels = self.transform_to_batch(x)
            out = self.model(inputs)['logits'].mean(1)
            return F.log_softmax(out), labels
        
        

