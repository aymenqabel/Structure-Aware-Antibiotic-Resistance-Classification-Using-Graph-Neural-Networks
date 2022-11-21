import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import pytorch_lightning as pl

from ..utils_seq import compute_metrics, load_sequences_COALA
from model import ESM_Model
from utils_esm import ESM_Dataset, idx_to_class


class Net(pl.LightningModule):   
    def __init__(self, seed = 0):
        super(Net, self).__init__()
        self.seed = seed
        self.lr = 8e-5
        self.batch_size = 1
        self.path_data = '../../data/splits1/'
        self.path_results = '../../results/'
        self.y_true = []
        self.y_pred = []
        self.names = []
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.model =  ESM_Model()   

    # Defining the forward pass    
    def forward(self, data):

        return self.model(data)
    
    def training_step(self, batch, batch_idx):
        self.model.name = False
        y_hat, y  = self(batch)
        loss = F.nll_loss(y_hat, y.view(-1))
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss
    
    def training_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch', avg_loss, prog_bar=True, batch_size=self.batch_size)
    
    def validation_step(self, batch, batch_nb):
        self.model.name = False
        y_hat, y  = self(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.val_accuracy.update(preds,  y.view(-1))
        loss = F.nll_loss(y_hat, y.view(-1))
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)
        self.log("val_acc", self.val_accuracy, prog_bar=True, batch_size=self.batch_size)
    
    def test_step(self, batch, batch_nb):
        self.model.name = True
        y_hat, y, name  = self(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.y_true += y.detach().tolist()
        self.y_pred += preds.detach().tolist()
        self.names += list(name)
        self.test_accuracy.update(preds, y.view(-1))
        loss = F.nll_loss(y_hat, y.view(-1))
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True, batch_size=self.batch_size)
        self.log("test_acc", self.test_accuracy, prog_bar=True, batch_size=self.batch_size)

    def test_epoch_end(self, outputs):
        self.y_true = [idx_to_class(x) for x in self.y_true]
        self.y_pred = [idx_to_class(x) for x in self.y_pred]
        self.name = f'ESM_650M_{self.seed}'
        return compute_metrics(self.y_true, self.y_pred, self.path_results, self.seed, 'ESM', names=self.names, name_result=self.name)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

    def prepare_data(self):
        self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test, self.names_test = load_sequences_COALA(self.path_data,seed=self.seed)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = ESM_Dataset(self.X_train, self.Y_train)
            self.val_dataset = ESM_Dataset(self.X_val, self.Y_val)
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = ESM_Dataset(self.X_test, self.Y_test, self.names_test)

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
    
    seed_everything(0)
    epochs = 30
    for seed in [0, 1, 2, 3, 4]:
        net = Net(seed = seed)
        checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/", save_top_k=2, monitor="val_acc", mode='max')
        trainer = pl.Trainer(accelerator='gpu',
                max_epochs=epochs,
                accumulate_grad_batches=32,
                callbacks=[checkpoint_callback],
                                )
        trainer.fit(net)
        out = trainer.test(ckpt_path='best')
