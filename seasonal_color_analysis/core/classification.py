import torch
import numpy as np


class CosineClassifier(torch.nn.Module):
  def __init__(self, n_classes: int):
    super().__init__()
    self.n_classes = n_classes
    self._temperature = torch.nn.Parameter(torch.rand(1))
    thetas = torch.arange(0, 2 * np.pi, 2 * np.pi / n_classes).view(-1, 1)
    cos = torch.cos(thetas)
    sin = torch.sin(thetas)
    self._points = torch.hstack([cos, sin]).unsqueeze(0)  # (1, landmarks, 2)
    self._cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    _x = x.unsqueeze(1)  # (batch, 1, 2)
    cos = self._cos(_x, self._points)  # cosine similarity with broadcasting
    logits = cos / self._temperature
    return logits


class SeasonEmbedder(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self._delta =  torch.nn.Parameter(torch.rand(1))
    self._embedder = torch.nn.Sequential(
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2)
        )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    w = self._embedder(x)
    return w * (1 + self._delta / torch.norm(w, dim=-1, keepdim=True))



class SeasonClassifier(torch.nn.Module):
  def __init__(self, n_classes: int):
    super().__init__()
    self._season_embedder = SeasonEmbedder()
    self._classifier = CosineClassifier(n_classes)

  def embedding(self, x: torch.Tensor) -> torch.Tensor:
    return self._season_embedder(x)

  def logits(self, embedding: torch.Tensor) -> torch.Tensor:
    return self._classifier(embedding)

  def theta(self, x: torch.Tensor) -> torch.Tensor:
    coordinates = self.embedding(x)
    return torch.atan2(coordinates[:, 1], coordinates[:, 0])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._classifier(self._season_embedder(x))