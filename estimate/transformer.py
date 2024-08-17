from torch import nn
from torch import optim
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2Model
import lightning as L
import torch
import numpy as np


class LinearData(Dataset):
  def __init__(self, data, n_time=50, scale=True):
    self.data = data
    self.subjects = data["subject"].unique()
    self.n_time = n_time
    self.y = [1.0 * (yi == "healthy") for yi in data["class"].values]
    self.x = np.array(data.iloc[:, 2:])
    if scale:
       self.x = (self.x - np.mean(self.x, axis=0)) / np.std(self.x, axis=0)


  def __len__(self):
    return len(self.subjects)

  def __getitem__(self, index):
    x = self.x[index, :].reshape((self.n_time, -1))
    return x.astype("float32"), torch.Tensor([self.y[index]])


class Transformer(nn.Module):
  def __init__(self, n_embd=144, n_positions=50, n_layer=6, n_class=2):
    super(Transformer, self).__init__()
    config = GPT2Config(n_embd=n_embd, n_positions=n_positions, n_layer=n_layer)
    self.backbone = GPT2Model(config)
    self.logits = nn.Linear(n_embd * n_positions, n_class - 1)

  def forward(self, x):
    z = self.backbone(inputs_embeds=x)
    return z, self.logits(z.last_hidden_state.view(x.shape[0], -1))


# define the LightningModule
class LitTransformer(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.acc = lambda y, p: ((y == 1) * (p > 0.0)).sum().item() + ((y == 0) * (p <= 0.0)).sum().item()

    def training_step(self, batch):
        x, y = batch
        _, p_hat = self.model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(p_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", self.acc(y, p_hat), on_epoch=True, reduce_fx=sum)
        return loss

    def validation_step(self, batch):
        x, y = batch
        _, p_hat = self.model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(p_hat, y)
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_acc", self.acc(y, p_hat), on_epoch=True, reduce_fx=sum)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-5)
        return optimizer
