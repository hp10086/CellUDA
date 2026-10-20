# -*- coding: utf-8 -*-
#
# Cell Classification Module
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essenfrom pathlib import Pathfrom pathlib import Path
# Modified from CellViT++ (https://github.com/TIO-IKIM/CellViT-plus-plus) — changes: hidden_dim may be given as a one-element list (as in our YAML configs); first element is used

from typing import Iterable, Union

import torch.nn as nn


class LinearClassifier(nn.Module):
    """Linear Classifier

    Args:
        embed_dim (int): Embedding dimension (input dimension)
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 100.
            A list/iterable is accepted; only its first element is used (single hidden layer).
        num_classes (int, optional): Number of output classes. Defaults to 2.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Union[int, Iterable[int]] = 100,
        num_classes: int = 2,
        drop_rate: float = 0,
    ):
        super(LinearClassifier, self).__init__()

        # Allow list/iterable but only use the first value to keep a single hidden layer
        if isinstance(hidden_dim, Iterable) and not isinstance(hidden_dim, (str, bytes)):
            hidden_dim = list(hidden_dim)[0]

        self.fc1 = nn.Linear(embed_dim, int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), num_classes)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
