from torch import nn
from torch import optim
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2Model
import lightning as L
import torch
import numpy as np


class ConceptData(Dataset):
  def __init__(self, data, concepts, n_time=50, scale=True):
    self.data = data
    self.subjects = data["subject"].unique()
    self.n_time = n_time
    self.concepts = np.array(concepts.iloc[:, 1:])
    self.n_concept = self.concepts.shape[1]
    self.y = [1.0 * (yi == "healthy") for yi in data["class"].values]
    self.x = np.array(data.iloc[:, 2:])
    if scale:
       self.x = (self.x - np.mean(self.x, axis=0)) / np.std(self.x, axis=0)


  def __len__(self):
    return len(self.subjects)

  def __getitem__(self, index):
    x = self.x[index, :].reshape((self.n_time, -1))
    c = torch.Tensor(self.concepts[index, :])
    return x.astype("float32"), c, torch.Tensor([self.y[index]])


class ConceptBottleneck(nn.Module):
  def __init__(self, n_embd=144, n_positions=50, n_layer=6, n_concept=25, n_class=2):
    super(ConceptBottleneck, self).__init__()
    self.n_concept = n_concept
    self.n_class = n_class
    config = GPT2Config(n_embd=n_embd, n_positions=n_positions, n_layer=n_layer)
    self.backbone = GPT2Model(config)
    self.concept = nn.Linear(n_embd * n_positions, self.n_concept)
    self.mlp = nn.Sequential(
       nn.Linear(self.n_concept, self.n_concept),
       nn.ReLU(),
       nn.Linear(self.n_concept, self.n_concept),
       nn.ReLU(),
       nn.Linear(self.n_concept, self.n_concept),
       nn.ReLU(),
       nn.Linear(self.n_concept, self.n_class - 1)
    )

  def forward(self, x):
    z = self.backbone(inputs_embeds=x)
    c = self.concept(z.last_hidden_state.view(x.shape[0], -1))
    return c, self.mlp(c)


# define the LightningModule
class LitConcept(L.LightningModule):
    def __init__(self, model, concept_hyper=.5):
        super().__init__()
        self.model = model
        self.concept_hyper = concept_hyper
        self.acc = lambda y, p: ((y == 1) * (p > 0.5)).sum().item() + ((y == 0) * (p <= 0.5)).sum().item()

    def training_step(self, batch):
        x, c, y = batch
        c_hat, y_hat = self.model(x)
        c_loss = nn.functional.binary_cross_entropy_with_logits(c_hat, c)
        y_loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        self.log("concept_loss", c_loss, on_epoch=True)
        self.log("task_loss", y_loss, on_epoch=True)
        self.log("train_acc", self.acc(y, y_hat), on_epoch=True, reduce_fx=sum)
        return c_loss + self.concept_hyper * y_loss

    def validation_step(self, batch):
        x, c, y = batch
        _, y_hat = self.model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_acc", self.acc(y, y_hat), on_epoch=True, reduce_fx=sum)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-5)
        return optimizer
