from torch import nn
from torch import optim
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2Model
import lightning as L
import torch
import numpy as np


class LinearData(Dataset):
  def __init__(self, data):
    self.data = data
    self.subjects = data["subject"].unique()

  def __len__(self):
    return len(self.subjects)

  def __getitem__(self, index):
    samples = self.data[self.data["subject"] == self.subjects[index]]
    x = samples.pivot(index="time", columns="taxon", values="Freq")
    y = [1.0 * (samples["class"].values[0] == "healthy")]
    return np.array(x).astype("float32"), torch.Tensor(y)


class Transformer(nn.Module):
  def __init__(self, n_embd=144, n_positions=50, n_layer=5, n_class=2):
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

    def training_step(self, batch, index):
        x, y = batch
        _, p_hat = self.model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(p_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

