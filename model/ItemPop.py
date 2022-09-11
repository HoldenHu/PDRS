# Author: Hengchang Hu
# Date: 03 July, 2020
# Description: The Model of top popularity recommendation without any metadata input
# -*- coding: utf-8 -*-

# Python imports
import torch
import numpy as np
import scipy.sparse as sp
from torch import nn


class ItemPop(nn.Module):
    def __init__(self, train_interaction_matrix: sp.dok_matrix, preloaded_item_ratings=None):
        super().__init__()
        """
        Simple popularity based recommender system
        preloaded_item_ratings for the model loaded in files: '*.npy'
        """
        self.__alias__ = "ItemPop"
        # Sum the occurences of each item to get is popularity, convert to array and 
        # lose the extra dimension
        if preloaded_item_ratings == None:
            self.item_ratings = np.array(train_interaction_matrix.sum(axis=0, dtype=int)).flatten()
        else:
            self.item_ratings = preloaded_item_ratings
        # full item's rating [97  1 15 ...  0  0  1]

    def forward(self, users, items) -> np.array:
        # returns the prediction score for each (user,item) pair in the input
        output_scores = [self.item_ratings[itemid] for itemid in items]
        return torch.from_numpy(np.array(output_scores))

    def get_alias(self):
        return self.__alias__