import torch
import random
import pandas as pd
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset


class UserItemRatingDataset(Dataset):
    
    def __init__(self, user_tensor, item_tensor, target_tensor, prior_tensor, post_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

        self.prior_tensor = prior_tensor
        self.post_tensor = post_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index], self.prior_tensor[index], self.post_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class DataSplitter():

    def __init__(self):
        self.ratings = self._load_rating()
        self._binalize()
        self.user_pool = set(self.ratings['new_uid'].unique())
        self.item_pool = set(self.ratings['new_mid'].unique())
        self.negatives = self._sample_negative()
        self.train_ratings, self.validation_ratings, self.test_ratings = self._split_data()

    def _load_rating(self):
        df = pd.read_csv(
            'Data/ml-1m/ratings.updated.dat',
            sep='::',
            header=None,
            names=['uid', 'mid', 'rating', 'timestamp'],
            engine='python'
            )

        user_id = df[['uid']].drop_duplicates()
        user_id['new_uid'] = np.arange(len(user_id))
        df = df.merge(user_id, on=['uid'])

        item_id = df[['mid']].drop_duplicates()
        item_id['new_mid'] = np.arange(len(item_id))
        df = df.merge(item_id, on=['mid'])

        df = df[['new_uid', 'new_mid', 'rating', 'timestamp']]
        return df

    def _binalize(self):
        self.ratings['rating'][self.ratings['rating'] > 0] = 1.0

    def _sample_negative(self):
        interact_status = \
            self.ratings.groupby('new_uid')['new_mid'].apply(set).reset_index().rename(columns={'new_mid': 'interacted_items'})
        interact_status['negative_items'] = \
            interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples_for_validation'] = \
            interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
        interact_status['negative_samples_for_test'] = \
            interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
        return interact_status[['new_uid', 'negative_items', 'negative_samples_for_validation', 'negative_samples_for_test']]

    def _split_data(self):
        self.ratings['timestamp_rank'] = \
            self.ratings.groupby(['new_uid'])['timestamp'].rank(method='first', ascending=False)
        test = self.ratings[self.ratings['timestamp_rank'] == 1]
        validation = self.ratings[self.ratings['timestamp_rank'] == 2]
        train = self.ratings[self.ratings['timestamp_rank'] > 2]
        return train[['new_uid', 'new_mid', 'rating']], validation[['new_uid', 'new_mid', 'rating']], test[['new_uid', 'new_mid', 'rating']]

    def make_evaluation_data(self, type):
        if type == 'test':
            ratings = pd.merge(self.test_ratings, self.negatives[['new_uid', 'negative_samples_for_test']], on='new_uid')
            ratings = ratings.rename(columns={'negative_samples_for_test': 'negative_samples'})
        elif type == 'validation':
            ratings = pd.merge(self.validation_ratings, self.negatives[['new_uid', 'negative_samples_for_validation']], on='new_uid')
            ratings = ratings.rename(columns={'negative_samples_for_validation': 'negative_samples'})
        users, items, negative_users, negative_items = [], [], [], []
        for row in ratings.itertuples():
            users.append(int(row.new_uid))
            items.append(int(row.new_mid))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.new_uid))
                negative_items.append(int(row.negative_samples[i]))
        return [torch.LongTensor(users), torch.LongTensor(items), torch.LongTensor(negative_users), torch.LongTensor(negative_items)]

    @property
    def n_user(self):
        return len(self.user_pool)

    @property
    def n_item(self):
        return len(self.item_pool)

    def make_train_loader(self, n_negative, batch_size):
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['new_uid', 'negative_items']], on='new_uid')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, n_negative))
        for row in train_ratings.itertuples():
            users.append(int(row.new_uid))
            items.append(int(row.new_mid))
            ratings.append(float(row.rating))
            for i in range(n_negative):
                users.append(int(row.new_uid))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    
    def load_postive_record_as_matrix(self):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users  = self.n_user
        num_items = self.n_item
        
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)

        for row in self.train_ratings.itertuples():
            user_id = int(row.new_uid)
            item_id = int(row.new_mid)
            mat[user_id, item_id] = 1.0
            
        return mat