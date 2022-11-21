"""
Model and utils function related to the loading of the ESM2 model
"""
import esm
import torch
import torch.nn as nn

class ESM_Model(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D() # 8 million parameters model
        self.batch_converter = self.alphabet.get_batch_converter()
        num_features = 320
        self.model.lm_head = nn.Linear(num_features, hidden_channels)
        self.name = False
    
    def transform_name(self, name):
        return "_".join(name[0].split("_")[:4])

    def transform_to_batch(self, data):
        new_data = [(self.transform_name(data[i].name), data[i].sequence[0]) for i in range(len(data.y))]
        seq_lenghts = [len(data[i].sequence[0]) for i in range(len(data.y))]
        new_batch = self.batch_converter(new_data)
        _, _, batch_tokens = new_batch
        y = data.y
        if self.name:
            return batch_tokens.long().cuda(), torch.LongTensor(y.cpu()).cuda(),seq_lenghts
        else:
            return batch_tokens.long().cuda(), torch.LongTensor(y.cpu()).cuda(), seq_lenghts

    def forward(self, g):
        inputs, _, seq_lenghts = self.transform_to_batch(g)
        out = self.model(inputs)['logits']
        x = out[0, 1:seq_lenghts[0]+1, :]
        for i in range(1, len(g.sequence)):
            x = torch.cat((x, out[i, 1:seq_lenghts[i]+1, :]), dim = 0)
        return x
        