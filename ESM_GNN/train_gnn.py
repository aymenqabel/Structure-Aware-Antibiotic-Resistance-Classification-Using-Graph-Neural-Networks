import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torchmetrics import Accuracy
import pytorch_lightning as pl

from ..gnn_test import GNN
from utils_seq import compute_metrics
from utils_graphs import feature_len, idx_to_class, load_graphs_COALA, select_features

class Net(pl.LightningModule):   
    def __init__(self, model_name, seed = 0, thresh = 10, n_layers = 6, hidden=64):
        super(Net, self).__init__()
        self.path_results = '../../results/'
        self.model_name = model_name
        self.y_true = []
        self.y_pred = []
        self.names = []
        self.lr = 1e-3
        self.batch_size = 128
        self.hparams.hidden = hidden
        self.hparams.n_layers = n_layers
        self.seed = seed
        self.hparams.features = ['phi', 'psi','rsa', 'asa', 'b_factor', 'ss','hbond_acceptors', 'hbond_donors', 'expasy']
        self.hparams.edges = [1,1,0,0,0,0]
        self.hparams.threshold = thresh
        self.criterion = nn.CrossEntropyLoss()
        self.input_dim = feature_len(self.hparams.features)
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.model = GNN(model_name, self.input_dim, self.hparams.hidden, 16, self.hparams.n_layers, 4)
        self.save_hyperparameters()

    # Defining the forward pass    
    def forward(self, data):
        x, edge_index, edge_att = select_features(data, features = self.hparams.features, edges = self.hparams.edges, threshold=self.hparams.threshold)
        return self.model(x, edge_index, edge_att,data.batch)
    
    def training_step(self, batch, batch_idx):
        y = batch.y
        y_hat = self(batch)
        loss = self.criterion(y_hat, y.view(-1))
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss
    
    def training_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch', avg_loss, prog_bar=True, batch_size=self.batch_size)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay=1e-4)
    
    def validation_step(self, batch, batch_nb):
        y = batch.y
        y_hat = self(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.val_accuracy.update(preds,  y.view(-1))
        loss = self.criterion(y_hat,  y.view(-1))
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)
        self.log("val_acc", self.val_accuracy, prog_bar=True, batch_size=self.batch_size)
    
    def test_step(self, batch, batch_nb):
        y = batch.y
        names = batch.name
        y_hat = self(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.y_true += y.detach().tolist()
        self.y_pred += preds.detach().tolist()
        self.names += names
        self.test_accuracy.update(preds, y.view(-1))
        loss = self.criterion(y_hat,  y.view(-1))
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True, batch_size=self.batch_size)
        self.log("test_acc", self.test_accuracy, prog_bar=True, batch_size=self.batch_size)
    
    def test_epoch_end(self, outputs):
        self.y_true = [idx_to_class(x) for x in self.y_true]
        self.y_pred = [idx_to_class(x) for x in self.y_pred]
        self.names = ["_".join(name[0].split("_")[:4]) for name in self.names]
        name = f'{self.model_name}_{self.seed}_{self.hparams.threshold}_{self.hparams.n_layers}_{self.hparams.hidden}_puregnn'
        return compute_metrics(self.y_true, self.y_pred, self.path_results, self.seed, 'GNN', names = self.names, name_result=name)

    def prepare_data(self):
        self.X_train, self.X_val, self.X_test = load_graphs_COALA(seed = self.seed)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = self.X_train
            self.val_dataset = self.X_test
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = self.X_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)




if __name__ == '__main__':
    
    from pytorch_lightning.callbacks import StochasticWeightAveraging

    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import seed_everything

    # seed_everything(0)
    epochs = 350
    model_name = 'GINE'
    seed = 0
    thresh = 10
    n_layers = 8
    hidden = 32
    for seed in [0,2,3,4]:
        net = Net(model_name, seed = seed, thresh = thresh, n_layers = n_layers, hidden=hidden)
        checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/", save_top_k=2, monitor="val_acc", mode='max')
        trainer = pl.Trainer(accelerator='gpu',
                            max_epochs=epochs,
                            accumulate_grad_batches=1,
                            callbacks=[checkpoint_callback]
                                )
        trainer.fit(net)
        out = trainer.test(ckpt_path='best')


