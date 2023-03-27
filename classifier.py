import torch.nn as nn
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl


class Classifier(pl.LightningModule):
    def __init__(self, lr: float = 0.003, task: str = 'classification'):
        super().__init__()
        self.lr = lr
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )
        self.loss_fn = nn.NLLLoss()
        self.train_acc = torchmetrics.Accuracy(task='multiclass')
        self.val_acc = torchmetrics.Accuracy(task='multiclass')

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        self.train_acc(y_hat.argmax(dim=1), y)
        self.log('train_acc', self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        self.val_acc(y_hat.argmax(dim=1), y)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
