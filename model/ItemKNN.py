#adapted from itemPOP & https://github.com/caserec/CaseRecommender/blob/master/caserec/recommenders/item_recommendation/itemknn.py
#Date: 19/10, 2020
#Description: Implements the ItemKNN model for recommendations


#Python imports
import torch
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import squareform, pdist

class ItemKNN():
    def __init__(self, k_neighbors, train_matrix, preloaded_variable_dict=None):
        """
        Simple item-based recommender system
        """
        self.__alias__ = "ItemKNN"
        
        self.train_matrix = train_matrix
        self.num_user = self.train_matrix.get_shape()[0]
        self.num_item = self.train_matrix.get_shape()[1]
        self.k_neighbors = k_neighbors
        self.train_matrix_to_array = self.train_matrix.toarray()
        self.train_matrix_dense = self.train_matrix.todense()
        
        if preloaded_variable_dict == None:
            #flatten interactions of each user
            self.user_hist = {}
            for user in range(0, self.num_user):
                self.user_hist[user] = np.flatnonzero(self.train_matrix_to_array[user])
            
            # Calculate distance matrix
            self.item_similarity_matrix = np.float32(squareform(pdist(self.train_matrix_dense.T, "cosine")))
            # Remove NaNs
            self.item_similarity_matrix[np.isnan(self.item_similarity_matrix)] = 1.0
            # transform distances in similarities. Values in matrix range from 0-1
            self.item_similarity_matrix = (self.item_similarity_matrix.max() - self.item_similarity_matrix) / self.item_similarity_matrix.max()
        
            self.similar_items = {}
            for item in range(0, self.num_item):
                self.similar_items[item] = sorted(range(len(self.item_similarity_matrix[item])), 
                    key=lambda k: -self.item_similarity_matrix[item][k])[0 : self.k_neighbors]
        else:
            # variables_dict = np.load(intermedia_file, allow_pickle=True).item()
            self.user_hist = preloaded_variable_dict['user_hist']
            self.item_similarity_matrix = preloaded_variable_dict['item_similarity_matrix']
            self.similar_items = preloaded_variable_dict['similar_items']

 
    def forward(self, users, items):
        # returns the prediction torch(score) for each (user,item) pair in the input
        intersect = {}
        output_scores = []

        for user in range(0, self.num_user):
            intersect[user] = {}

        for (u,i) in zip(users, items):
            u = int(u)
            i = int(i)
            intersect[u][i] = 0.0
            for hist_item in self.user_hist[u]:
                if hist_item in self.similar_items[i]:
                    intersect[u][i] = intersect[u][i] + self.item_similarity_matrix[i][hist_item]
            output_scores.append(intersect[u][i])
        
        return torch.from_numpy(np.array(output_scores))

    def get_alias(self):
        return self.__alias__