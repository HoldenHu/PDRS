# Author: Hengchang Hu
# Date: 30 Nov, 2020
# Description: to read the data as a whole, and split them into train/test in different manners.

import os
import time
import torch
import json
import random
import scipy.sparse as sp
import pandas as pd
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

# from utility.test_util import get_knowledgeScore

class UserItemRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, rating_tensor, priork_tensor, targetk_tensor, itemk_tensor, valid_priork_num, valid_targetk_num, valid_itemk_num):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.rating_tensor = rating_tensor

        self.priork_tensor = priork_tensor # list<np.LongTensor>
        self.targetk_tensor = targetk_tensor # list<np.LongTensor>
        self.itemk_tensor = itemk_tensor

        self.valid_priork_num = valid_priork_num # list<int>
        self.valid_targetk_num = valid_targetk_num # list<int>
        self.valid_itemk_num = valid_itemk_num # list<int>


    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.rating_tensor[index], self.priork_tensor[index], self.targetk_tensor[index], self.itemk_tensor[index], self.valid_priork_num[index], self.valid_targetk_num[index], self.valid_itemk_num[index]

    def __len__(self):
        return self.user_tensor.size(0)

class BPRDataset(Dataset):
    def __init__(self, user_tensor, item_pos_tensor, item_neg_tensor):
        self.user_tensor = user_tensor
        self.item_pos_tensor = item_pos_tensor
        self.item_neg_tensor = item_neg_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_pos_tensor[index], self.item_neg_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

class KnowledgeLinkingDataset(Dataset):
    def __init__(self, fromk_tensor, tok_tensor, score_tensor):
        self.fromk_tensor = fromk_tensor
        self.tok_tensor = tok_tensor
        self.score_tensor = score_tensor

    def __getitem__(self, index):
        return self.fromk_tensor[index], self.tok_tensor[index], self.score_tensor[index]

    def __len__(self):
        return self.fromk_tensor.size(0)

class DataSplitter():
    def __init__(self, config, task = 'KDRS', load_from_cache = True):
        '''
        Read from config =>
        task: KDRS/ KLP/ RS

        data_ratings_path: 
            user<int> - item<int> - rating<int> - timestamp<int>
        data_userk_path:
            user<int> - priork<list<int>> - targetk<list<int>>
        data_itemk_path:
            item<int> - containk<list<int>>
        data_klinking_path:
            fromk<int> - towardk<int> - score<float: 0~1>

        negative_strategy:
            random: random choose from negative items
            cluster: TODO
        split_strategy:
            warm-start
            cold-start
        '''

        self.random_seed = 2
        data_type = config.get('DATA', 'data_type')
        data_folder = config.get('DATA', 'data_folder')
        data_ratings_file = config.get('DATA', 'data_ratings_file')
        data_userk_file = config.get('DATA', 'data_userk_file')
        data_itemk_file = config.get('DATA', 'data_itemk_file')
        data_klinking_file = config.get('DATA', 'data_klinking_file')
        knowledge_word_file = config.get('DATA', 'knowledge_word_file')

        data_ratings_path = os.path.join(data_folder,data_type,data_ratings_file)
        data_userk_path = os.path.join(data_folder,data_type,data_userk_file)
        data_itemk_path = os.path.join(data_folder,data_type,data_itemk_file)
        data_klinking_path = os.path.join(data_folder,data_type,data_klinking_file)
        knowledge_word_path = os.path.join(data_folder,data_type,knowledge_word_file)

        negative_strategy = config.get('DATA', 'negative_strategy') # defaultly, 'random'
        cluster_num = config.getint('DATA', 'cluster_num') # defaultly, 20
        split_strategy = config.get('DATA', 'split_strategy') # defaultly, 'cold-start'
        n_neg_train = config.getint('DATA', 'n_neg_train') # defaultly, 4
        n_neg_test = config.getint('DATA', 'n_neg_test') # defaultly, 99
        n_kneg_test = config.getint('DATA', 'n_kneg_test') # defaultly, 99
        bz = config.getint('MODEL', 'batch_size') # defaultly, 128
        kbz = config.getint('MODEL', 'kbatch_size') # defaultly, 64
        print('>> [Data Preparation]<Data pre-processing strategy> negative_strategy: {}, split_strategy: {}, n_neg_train: {}, n_neg_test: {}'.format(negative_strategy, split_strategy, n_neg_train, n_neg_test))

        cache_folder = os.path.join(data_folder,data_type,'cache')
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
        if negative_strategy == 'cluster':
            cache_folder = os.path.join(cache_folder,negative_strategy+str(cluster_num)+'.'+split_strategy+'.bz'+str(bz)+'.kbz'+str(kbz))
        else: # random
            cache_folder = os.path.join(cache_folder,negative_strategy+'.'+split_strategy+'.bz'+str(bz)+'.kbz'+str(kbz))
        
        ##################################### Pure RS
        if task == 'RS':
            # 不包含任何knowledge信息的pale interaction dataset
            if os.path.exists(cache_folder) and load_from_cache:
                print(">> [Data Preparation] RS task - Load data from path", cache_folder)
                intermedia_file = os.path.join(cache_folder,'variable.npy')
                variables_dict = np.load(intermedia_file, allow_pickle=True).item()
                self.user_pool = variables_dict['user_pool']
                self.item_pool = variables_dict['item_pool']
                self.user_rating_dict = variables_dict['user_rating_dict']
                self.train_dataset = variables_dict['train_dataset']
                self.validation_dataset = variables_dict['validation_dataset']
                self.test_dataset = variables_dict['test_dataset']
                self.pos_train_dataset = variables_dict['pos_train_dataset']

                self.validation_loader = torch.load(os.path.join(cache_folder,'validation_loader.pt'))
                self.test_loader = torch.load(os.path.join(cache_folder,'test_loader.pt'))
                self.train_loader = torch.load(os.path.join(cache_folder,'train_loader.pt'))
                print(">> [Data Preparation] (with file loading) Succeffully generated full(pos+neg) dataset into train dataset with {}, valid dataset with {}, test dataset with {}".format(len(self.train_dataset), len(self.validation_dataset), len(self.test_dataset)))
            else:
                self.rating_df = pd.read_csv(data_ratings_path,index_col=0)
                print(">> [Data Preparation] Loaded alll data from files")
                self.all_dataset_len = len(self.rating_df)
                
                self.user_pool = set(self.rating_df['user']) # {0,...}
                self.item_pool = set(self.rating_df['item'])
                self._user_newid_dict = self._create_cat_dict(self.user_pool)
                self._item_newid_dict = self._create_cat_dict(self.item_pool)
                # 只把rating中的user id和item id重新分配
                self._resample_ratings_id()
                print(">> [Data Preparation] Finished resampling user and item ids.")
                self.user_rating_dict = self._make_user_rating_dict(self.rating_df) # {0:[(3, 19), (7, 10000), (6, 200000),],...}
                
                # 把每个user对应的negative sample pool先确定了, train, test, valid都是从中随机生成
                self.user_negative_pool = self._create_negative_pool(negative_strategy, knowledge_word_path, cluster_num) # Usage: user_negative_pool[userid] => [cids,]
                self.pos_train_dataset, self.pos_validation_dataset, self.pos_test_dataset = self._split_data(split_strategy)
                print(">> [Data Preparation] Splitted postive dataset into train dataset with {}, valid dataset with {}, test dataset with {}".format(len(self.pos_train_dataset), len(self.pos_validation_dataset), len(self.pos_test_dataset)))
                
                # 加上negative sample
                self.train_dataset = self._add_negative_sample(self.pos_train_dataset, n_neg_train) # [(int(uid),int(itemid),label),...]
                self.validation_dataset = self._add_negative_sample(self.pos_validation_dataset, n_neg_test)
                self.test_dataset = self._add_negative_sample(self.pos_test_dataset, n_neg_test)
                print(">> [Data Preparation] Succeffully generated full(pos+neg) dataset into train dataset with {}, valid dataset with {}, test dataset with {}".format(len(self.train_dataset), len(self.validation_dataset), len(self.test_dataset)))
                
                self.validation_loader = self.make_Y_validation_loader(n_neg_test+1, enriched=True)
                self.test_loader = self.make_Y_test_loader(n_neg_test+1, enriched=True)
                self.train_loader = self.make_Y_train_loader(bz, enriched=True)
        ##################################### KLP
        elif task == 'KLP':
            if os.path.exists(cache_folder) and load_from_cache:
                print(">> [Data Preparation] KLP task - Load data from path", cache_folder)
                intermedia_file = os.path.join(cache_folder,'variable.npy')
                variables_dict = np.load(intermedia_file, allow_pickle=True).item()
                self.knowledge_pool = variables_dict['knowledge_pool']
                self.klinking_test_dataset = variables_dict['klinking_test_dataset']
                self.klinking_train_dataset = variables_dict['klinking_train_dataset']
                self.klinking_pre_df = variables_dict['klinking_pre_df']

                self.k_test_loader = torch.load(os.path.join(cache_folder,'k_test_loader.pt'))
                self.k_train_loader = torch.load(os.path.join(cache_folder,'k_train_loader.pt'))
                print(">> [Data Preparation] (with file loading) Succeffully generated full knowlege linking dataset into train dataset with {}, test dataset with {}".format(len(self.klinking_train_dataset), len(self.klinking_test_dataset)))
            else:  
                self.klinking_pre_df = pd.read_csv(data_klinking_path,index_col=0)
                print(">> [Data Preparation] Loaded knowledge linking data from files")
                self.knowledge_pool = set(self.klinking_pre_df['fromk']) | set(self.klinking_pre_df['towardk'])
                self._knowledge_newid_dict = self._create_cat_dict(self.knowledge_pool)
                self._resample_kl_id()
                print(">> [Data Preparation] Finished resampling knowledge ids.")
                self.klinking_test_dataset, self.klinking_train_dataset = self._split_k_linking()
                print(">> [Data Preparation] Succeffully generated full knowlege linking dataset into train dataset with {}, test dataset with {}".format(len(self.klinking_train_dataset), len(self.klinking_test_dataset)))
                self.k_test_loader = self.make_K_test_loader(n_kneg_test+1)
                self.k_train_loader = self.make_K_train_loader(kbz)
        
        ##################################### KDRS
        else: # task == KDRS
            # 包括knowledge信息的enriched dataset
            if os.path.exists(cache_folder) and load_from_cache:
                print(">> [Data Preparation] KDRS task - Load data from path", cache_folder)
                intermedia_file = os.path.join(cache_folder,'variable.npy')
                variables_dict = np.load(intermedia_file, allow_pickle=True).item()
                self.user_pool = variables_dict['user_pool']
                self.item_pool = variables_dict['item_pool']
                self.knowledge_pool = variables_dict['knowledge_pool']
                self.user_rating_dict = variables_dict['user_rating_dict']
                self.user_priork_dict = variables_dict['user_priork_dict']
                self.user_targetk_dict = variables_dict['user_targetk_dict']
                self.item_containk_dict = variables_dict['item_containk_dict']
                self.train_dataset = variables_dict['train_dataset']
                self.validation_dataset = variables_dict['validation_dataset']
                self.test_dataset = variables_dict['test_dataset']
                self.klinking_test_dataset = variables_dict['klinking_test_dataset']
                self.klinking_train_dataset = variables_dict['klinking_train_dataset']

                self.pos_train_dataset = variables_dict['pos_train_dataset']
                self.klinking_pre_df = variables_dict['klinking_pre_df']

                self.validation_loader = torch.load(os.path.join(cache_folder,'validation_loader.pt'))
                self.test_loader = torch.load(os.path.join(cache_folder,'test_loader.pt'))
                self.train_loader = torch.load(os.path.join(cache_folder,'train_loader.pt'))
                self.k_test_loader = torch.load(os.path.join(cache_folder,'k_test_loader.pt'))
                self.k_train_loader = torch.load(os.path.join(cache_folder,'k_train_loader.pt'))
                print(">> [Data Preparation] (with file loading) Succeffully generated full(pos+neg) dataset into train dataset with {}, valid dataset with {}, test dataset with {}".format(len(self.train_dataset), len(self.validation_dataset), len(self.test_dataset)))
            else:
                self.rating_df = pd.read_csv(data_ratings_path,index_col=0)
                self.userk_df = pd.read_csv(data_userk_path,index_col=0,converters={'priork': eval,'targetk': eval})
                self.itemk_df = pd.read_csv(data_itemk_path,index_col=0,converters={'containk': eval})
                self.klinking_pre_df = pd.read_csv(data_klinking_path,index_col=0)
                print(">> [Data Preparation] Loaded alll data from files")
                self.all_dataset_len = len(self.rating_df)

                self.user_pool = set(self.rating_df['user']) # {0,...}
                self.item_pool = set(self.rating_df['item'])
                self.knowledge_pool = self._get_knowledgeset()
                self._user_newid_dict = self._create_cat_dict(self.user_pool)
                self._item_newid_dict = self._create_cat_dict(self.item_pool)
                self._knowledge_newid_dict = self._create_cat_dict(self.knowledge_pool)

                # 把所有数据中的user id和item id 和knowledge id重新分配
                self._resample_id()
                print(">> [Data Preparation] Finished resampling user,item, and knowledge ids.")
                # 创建easy-access的数据格式来使用
                self.user_rating_dict = self._make_user_rating_dict(self.rating_df) # {0:[(3, 19), (7, 10000), (6, 200000),],...}
                self.user_priork_dict, self.user_targetk_dict, self.item_containk_dict = self._make_kmapping_dict(self.itemk_df, self.userk_df)
                # self.klinking_pre_dict = self._make_klinking_dict() # {(1,2):0.54,}

                # 把每个user对应的negative sample pool先确定了, train, test, valid都是从中随机生成
                self.user_negative_pool = self._create_negative_pool(negative_strategy, knowledge_word_path, cluster_num) # Usage: user_negative_pool[userid] => [cids,]
                self.pos_train_dataset, self.pos_validation_dataset, self.pos_test_dataset = self._split_data(split_strategy)
                print(">> [Data Preparation] Splitted postive dataset into train dataset with {}, valid dataset with {}, test dataset with {}".format(len(self.pos_train_dataset), len(self.pos_validation_dataset), len(self.pos_test_dataset)))
                
                # 加上negative sample
                self.train_dataset = self._add_negative_sample(self.pos_train_dataset, n_neg_train) # [(int(uid),int(itemid),label),...]
                self.validation_dataset = self._add_negative_sample(self.pos_validation_dataset, n_neg_test)
                self.test_dataset = self._add_negative_sample(self.pos_test_dataset, n_neg_test)
                self.klinking_test_dataset, self.klinking_train_dataset = self._split_k_linking()
                print(">> [Data Preparation] Succeffully generated full(pos+neg) dataset into train dataset with {}, valid dataset with {}, test dataset with {}".format(len(self.train_dataset), len(self.validation_dataset), len(self.test_dataset)))
                
                os.makedirs(cache_folder)
                saving_dict = {
                    'user_pool':self.user_pool,
                    'item_pool':self.item_pool,
                    'knowledge_pool':self.knowledge_pool,
                    'user_rating_dict':self.user_rating_dict,
                    'user_priork_dict':self.user_priork_dict,
                    'user_targetk_dict':self.user_targetk_dict,
                    'item_containk_dict':self.item_containk_dict,
                    'train_dataset':self.train_dataset,
                    'validation_dataset':self.validation_dataset,
                    'test_dataset':self.test_dataset,
                    'klinking_test_dataset':self.klinking_test_dataset,
                    'klinking_train_dataset':self.klinking_train_dataset,

                    'klinking_pre_df': self.klinking_pre_df,
                    'pos_train_dataset': self.pos_train_dataset
                }
                intermedia_file = os.path.join(cache_folder,'variable.npy')
                np.save(intermedia_file,saving_dict)

                self.validation_loader = self.make_Y_validation_loader(n_neg_test+1, enriched=True)
                self.test_loader = self.make_Y_test_loader(n_neg_test+1, enriched=True)
                self.train_loader = self.make_Y_train_loader(bz, enriched=True)
                self.k_test_loader = self.make_K_test_loader(n_kneg_test+1)
                self.k_train_loader = self.make_K_train_loader(kbz)
                torch.save(self.validation_loader,os.path.join(cache_folder,'validation_loader.pt'))
                torch.save(self.test_loader,os.path.join(cache_folder,'test_loader.pt'))
                torch.save(self.train_loader,os.path.join(cache_folder,'train_loader.pt'))
                torch.save(self.k_test_loader,os.path.join(cache_folder,'k_test_loader.pt'))
                torch.save(self.k_train_loader,os.path.join(cache_folder,'k_train_loader.pt'))

                print(">> [Data Preparation] Intermedia and data loader file saved, while the file do not exist.")
            #####################################

    def _make_user_rating_dict(self,rating_df):
        user_rating_dict = {}
        for i in rating_df.index:
            user = rating_df.at[i,'user']
            item = rating_df.at[i,'item']
            timestamp = rating_df.at[i,'timestamp']
            if user in user_rating_dict:
                user_rating_dict[user].append((item,timestamp))
            else:
                user_rating_dict[user] = [(item,timestamp)]
        # 按照timestamp排序
        for user in user_rating_dict:
            hist_list = user_rating_dict[user]
            user_rating_dict[user] = sorted(hist_list, key = lambda kv:(kv[1], kv[0]))
        return user_rating_dict


    def _make_kmapping_dict(self,itemk_df,userk_df):
        item_containk_dict = {}
        user_priork_dict = {}
        user_targetk_dict = {}

        for i in itemk_df.index:
            item = itemk_df.at[i,'item']
            containk = itemk_df.at[i,'containk']
            item_containk_dict[item] = containk

        for i in userk_df.index:
            user = userk_df.at[i,'user']
            priork = userk_df.at[i,'priork']
            targetk = userk_df.at[i,'targetk']
            user_priork_dict[user] = priork
            user_targetk_dict[user] = targetk
        return user_priork_dict, user_targetk_dict, item_containk_dict


    def _make_klinking_dict(self):
        linking_dict = {}
        for i in self.klinking_pre_df.index:
            linking_dict[(self.klinking_pre_df.at[i,'fromk'], self.klinking_pre_df.at[i,'towardk'])] = self.klinking_pre_df.at[i,'score']
        return linking_dict


    def _get_knowledgeset(self):
        knowledge_set = set()
        for k_list in self.userk_df['priork']:
            for k in k_list:
                knowledge_set.add(k)
        for k_list in self.userk_df['targetk']:
            for k in k_list:
                knowledge_set.add(k)
        for k_list in self.itemk_df['containk']:
            for k in k_list:
                knowledge_set.add(k)
        for k in self.klinking_pre_df['fromk']:
            knowledge_set.add(k)
        for k in self.klinking_pre_df['towardk']:
            knowledge_set.add(k)
        
        return knowledge_set


    def _create_cat_dict(self, cat_set):
        cat_dict = {}
        for cat in cat_set:
            id = len(cat_dict)
            cat_dict[cat] = id
        return cat_dict


    def _resample_ratings_id(self):
        # resample id in rating_df
        for i in self.rating_df.index:
            self.rating_df.at[i,'user'] = self._user_newid_dict[self.rating_df.at[i,'user']]
            self.rating_df.at[i,'item'] = self._item_newid_dict[self.rating_df.at[i,'item']]
        # resample id in user pool and item pool
        user_pool, item_pool = self.user_pool, self.item_pool
        self.user_pool = {self._user_newid_dict[user] for user in user_pool}
        self.item_pool = {self._item_newid_dict[item] for item in item_pool}
        return

    def _resample_kl_id(self):
        # resample id in knowledge linking
        for i in self.klinking_pre_df.index:
            self.klinking_pre_df.at[i,'fromk'] = self._knowledge_newid_dict[self.klinking_pre_df.at[i,'fromk']]
            self.klinking_pre_df.at[i,'towardk'] = self._knowledge_newid_dict[self.klinking_pre_df.at[i,'towardk']]
        knowledge_pool = self.knowledge_pool
        self.knowledge_pool = {self._knowledge_newid_dict[k] for k in knowledge_pool}
        return

    def _resample_id(self):
        # resample id in rating_df
        for i in self.rating_df.index:
            self.rating_df.at[i,'user'] = self._user_newid_dict[self.rating_df.at[i,'user']]
            self.rating_df.at[i,'item'] = self._item_newid_dict[self.rating_df.at[i,'item']]
        # resample id in userk_df
        for i in self.userk_df.index:
            self.userk_df.at[i,'user'] = self._user_newid_dict[self.userk_df.at[i,'user']]
            prior_k_list = self.userk_df.at[i,'priork']
            target_k_list = self.userk_df.at[i,'targetk']
            self.userk_df.at[i,'priork'] = [self._knowledge_newid_dict[k] for k in prior_k_list]
            self.userk_df.at[i,'targetk'] = [self._knowledge_newid_dict[k] for k in target_k_list]
        # resample id in itemk_df,因为有一些item在rating中没有，但在itemk中有，因此要删减
        for i in self.itemk_df.index:
            new_itemid = self._item_newid_dict.get(self.itemk_df.at[i,'item'])
            if new_itemid == None:
                # delete the row
                self.itemk_df.drop([i])
            else:
                self.itemk_df.at[i,'item'] = new_itemid
                contain_k_list = self.itemk_df.at[i,'containk']
                self.itemk_df.at[i,'containk'] = [self._knowledge_newid_dict[k] for k in contain_k_list]
        # resample id in knowledge linking
        for i in self.klinking_pre_df.index:
            self.klinking_pre_df.at[i,'fromk'] = self._knowledge_newid_dict[self.klinking_pre_df.at[i,'fromk']]
            self.klinking_pre_df.at[i,'towardk'] = self._knowledge_newid_dict[self.klinking_pre_df.at[i,'towardk']]
        # resample id in user pool and item pool, knowledge pool
        user_pool, item_pool, knowledge_pool = self.user_pool, self.item_pool, self.knowledge_pool
        self.user_pool = {self._user_newid_dict[user] for user in user_pool}
        self.item_pool = {self._item_newid_dict[item] for item in item_pool}
        self.knowledge_pool = {self._knowledge_newid_dict[k] for k in knowledge_pool}
        return


    def _create_negative_pool(self, negative_strategy, knowledge_word_path, cluster_num):
        '''
        negative_strategy: 'random'/'cluster'
        '''
        user_negative_pool_dict = {}

        if negative_strategy == 'cluster':
            import spacy
            import en_core_web_lg
            from sklearn.cluster import KMeans
            lg_nlp = en_core_web_lg.load()
            
            knowledge_dict = json.load(open(knowledge_word_path))
            kvector_dict = {}
            for kid in knowledge_dict.keys():
                kuword = knowledge_dict[kid]
                kid = self._knowledge_newid_dict[int(kid)]
                # kid = self._knowledge_newid_dict.get(int(kid))
                # if kid == None:
                #     # print("[TEST] ERROR")
                #     continue
                v = lg_nlp(u'%s' %kuword).vector
                kvector_dict[kid] = v
            
            itemid_list = []
            feature_matrix = []
            for i in self.itemk_df.index:
                itemid = self.itemk_df.at[i,'item']
                item_matrix = []
                containk_list = self.itemk_df.at[i,'containk']
                for kid in containk_list:
                    v = kvector_dict[kid]
                    item_matrix.append(v)
                item_matrix = np.array(item_matrix)
                item_vector = np.mean(item_matrix, axis=0)
                itemid_list.append(itemid)
                feature_matrix.append(item_vector)
            feature_matrix = np.array(feature_matrix)

            kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(feature_matrix)
            labels_list = kmeans.labels_
            item_labels_dict = {}
            for i in range(len(itemid_list)):
                itemid = itemid_list[i]
                label = labels_list[i]
                item_labels_dict[itemid] = label
            print(">>> [Data Preprocessing] Finished item clustering.")

            for each_userid in self.user_rating_dict:
                user_pos_label = set()
                item_hist = [i[0] for i in self.user_rating_dict[each_userid]]
                for i in item_hist:
                    user_pos_label.add(i)
                user_negative_pool_dict[each_userid] = [ i for i in self.item_pool if item_labels_dict[i] not in user_pos_label ]
        else: # 'random'
            for each_userid in self.user_rating_dict:
                item_hist = [i[0] for i in self.user_rating_dict[each_userid]]
                user_negative_pool_dict[each_userid] = [ i for i in self.item_pool if i not in item_hist ]
        
        return user_negative_pool_dict
    

    def _split_data(self, split_strategy):
        user_hist_dict = {} #{0:[1,2,3],}
        for each_userid in self.user_rating_dict:
            item_hist = [i[0] for i in self.user_rating_dict[each_userid]]
            user_hist_dict[each_userid] = item_hist
        
        pick_list = list(user_hist_dict.keys())
        r = random.random
        random.seed(self.random_seed)
        random.shuffle(pick_list,random=r)

        valid_ratio, test_ratio = 0.05, 0.15
        train_pos, valid_pos, test_pos = [], [], [] # 都是[(userid,itemid,label),..]
        valid_size= int(valid_ratio*self.all_dataset_len)
        test_size = int(test_ratio*self.all_dataset_len)
        train_size = self.all_dataset_len - (valid_size+test_size)
        if split_strategy == 'warm-start':
            while valid_size!= 0 and test_size!= 0:
                for userid in pick_list:
                    if len(user_hist_dict[userid]) > 1:
                        itemid = user_hist_dict[userid].pop()
                        if test_size != 0: # add into test set
                            test_pos.append((userid,itemid,1))
                            test_size = test_size - 1
                        else: # add into valid set
                            valid_pos.append((userid,itemid,1))
                            valid_size = valid_size - 1
            # 剩下的放入train中
            for userid in user_hist_dict:
                item_list = user_hist_dict[userid]
                for itemid in item_list:
                    train_pos.append((userid,itemid,1))
        elif split_strategy == 'itemcold-start':  # item cold-start
            valid_n_item, test_n_item = int(self.n_item*valid_ratio), int(self.n_item*test_ratio)
            item_pick_list = list(self.item_pool)
            r = random.random
            random.seed(self.random_seed)
            random.shuffle(item_pick_list,random=r)

            valid_itemid_list = item_pick_list[:valid_n_item]
            test_itemid_list = item_pick_list[valid_n_item:valid_n_item+test_n_item]
            train_itemid_list = item_pick_list[valid_n_item+test_n_item:]
            for userid in user_hist_dict:
                item_list = user_hist_dict[userid]
                for itemid in item_list:
                    if itemid in valid_itemid_list:
                        valid_pos.append((userid,itemid,1))
                    elif itemid in test_itemid_list:
                        test_pos.append((userid,itemid,1))
                    else: # train_itemid_list
                        train_pos.append((userid,itemid,1))

        else: # user cold-start
            valid_n_user, test_n_user = int(self.n_user*valid_ratio), int(self.n_user*test_ratio)
            valid_userid_list = pick_list[:valid_n_user]
            test_userid_list = pick_list[valid_n_user:valid_n_user+test_n_user]
            train_userid_list = pick_list[valid_n_user+test_n_user:]

            # 直接按照userid横切
            for userid in valid_userid_list:
                item_list = user_hist_dict[userid]
                for itemid in item_list:
                    valid_pos.append((userid,itemid,1))
            for userid in test_userid_list:
                item_list = user_hist_dict[userid]
                for itemid in item_list:
                    test_pos.append((userid,itemid,1))
            for userid in train_userid_list:
                item_list = user_hist_dict[userid]
                for itemid in item_list:
                    train_pos.append((userid,itemid,1))

        return train_pos, valid_pos, test_pos


    def _split_k_linking(self):
        test_set, train_set = [],[]
        test_ratio = 0.05
        pick_list = list(self.klinking_pre_df.index)
        test_num = int(test_ratio*len(pick_list))
        r = random.random
        random.seed(self.random_seed)
        random.shuffle(pick_list,random=r)
        for i in pick_list:
            fromk = self.klinking_pre_df.at[i,'fromk']
            tok = self.klinking_pre_df.at[i,'towardk']
            score = self.klinking_pre_df.at[i,'score']
            if test_num != 0:
                test_set.append((fromk,tok,score))
                test_num = test_num - 1
            else:
                train_set.append((fromk,tok,score))
        return test_set,train_set


    def _add_negative_sample(self, pos_dataset, n_neg):
        '''
        pos_dataset: [(userid, itemid, label)]
        '''
        added_dataset = []
        for row in pos_dataset:
            userid = row[0]
            pos_item = row[1]
            added_dataset.append((userid, pos_item, 1))
            negative_pool = self.user_negative_pool[userid]
            neg_items = random.sample(negative_pool, n_neg)
            for each_neg_item in neg_items:
                added_dataset.append((userid, each_neg_item, 0))
        return added_dataset

    @property
    def n_user(self):
        return len(self.user_pool)

    @property
    def n_item(self):
        return len(self.item_pool)
    
    @property
    def n_knowledge(self):
        return len(self.knowledge_pool)


    def load_train_rating_matrix(self):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users  = self.n_user
        num_items = self.n_item
        
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)

        for row in self.pos_train_dataset:
            user_id = int(row[0])
            item_id = int(row[1])
            mat[user_id, item_id] = 1.0
            
        return mat

    def load_train_knowledge_matrix(self):
        num_k = self.n_knowledge

        mat = sp.dok_matrix((num_k+1, num_k+1), dtype=np.float32)
        for i in self.klinking_pre_df.index:
            fromk = self.klinking_pre_df.at[i,'fromk']
            towardk = self.klinking_pre_df.at[i,'towardk']
            score = self.klinking_pre_df.at[i,'score']
            mat[fromk, towardk] = score
        return mat


    def __padding_list(self,v_list,padding_value=0):
        '''
        v_list: list with variable length: e.g., [[1,2,3], [2,4,5,6],[2]]
        return:
             array([[1, 2, 3, 0],
                    [2, 4, 5, 6],
                    [2, 0, 0, 0]])
        '''
        x = np.array(v_list)
        max_length = max(len(row) for row in x)
        # x_padded = torch.LongTensor([row + [0] * (max_length - len(row)) for row in x])
        x_padded = [row + [0] * (max_length - len(row)) for row in x]
        return x_padded

    def make_Y_validation_loader(self, batch_size, enriched = True):
        '''
        batch_size: should be n_test_neg_num + 1
        enriched: if False, then store None into the user_priorks, user_targetks, and item_containks, meanwhile the valid_priork_num, valid_targetk_num, valid_itemk_num will be zero
                  if True, store the knowledge information into the dataset
        '''
        users, items, ratings, user_priorks, user_targetks, item_containks = [], [], [], [], [],[]
        valid_priork_num, valid_targetk_num, valid_itemk_num = [],[],[] # 存储每一组data实际使用到的k数量
        for row in self.validation_dataset:
            u = int(row[0])
            i = int(row[1])
            users.append(u)
            items.append(i)
            ratings.append(float(row[2]))
            if enriched:
                priork = self.user_priork_dict.get(u)
                targetk = self.user_targetk_dict.get(u)
                itemk = self.item_containk_dict.get(i)
                if priork == None:
                    print(">> [Warning] Skip one data sample cuz priork key value error.")
                    continue
                if targetk == None:
                    print(">> [Warning] Skip one data sample cuz targetk key value error.")
                    continue
                if itemk == None:
                    print(">> [Warning] Skip one data sample cuz itemk key value error.")
                    continue
                user_priorks.append(priork)
                valid_priork_num.append(len(priork))
                user_targetks.append(targetk)
                valid_targetk_num.append(len(targetk))
                item_containks.append(itemk)
                valid_itemk_num.append(len(itemk))
            else:
                user_priorks.append([])
                valid_priork_num.append(0)
                user_targetks.append([])
                valid_targetk_num.append(0)
                item_containks.append([])
                valid_itemk_num.append(0)
        
        # pad 0 into the untidy list
        user_priorks = self.__padding_list(user_priorks)
        user_targetks = self.__padding_list(user_targetks)
        item_containks = self.__padding_list(item_containks)

        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        rating_tensor=torch.FloatTensor(ratings),
                                        priork_tensor=torch.LongTensor(user_priorks),
                                        targetk_tensor=torch.LongTensor(user_targetks),
                                        itemk_tensor=torch.LongTensor(item_containks),
                                        valid_priork_num=valid_priork_num,
                                        valid_targetk_num=valid_targetk_num,
                                        valid_itemk_num=valid_itemk_num)
        return DataLoader(dataset, batch_size = batch_size, shuffle=False, num_workers=0)
    
    def make_Y_test_loader(self, batch_size, enriched = True):
        '''
        batch_size: should be n_test_neg_num + 1
        enriched: if False, then store None into the user_priorks, user_targetks, and item_containks, meanwhile the valid_priork_num, valid_targetk_num, valid_itemk_num will be zero
                  if True, store the knowledge information into the dataset
        '''
        users, items, ratings, user_priorks, user_targetks, item_containks = [], [], [], [], [],[]
        valid_priork_num, valid_targetk_num, valid_itemk_num = [],[],[] # 存储每一组data实际使用到的k数量
        for row in self.test_dataset:
            u = int(row[0])
            i = int(row[1])
            users.append(u)
            items.append(i)
            ratings.append(float(row[2]))
            if enriched:
                priork = self.user_priork_dict.get(u)
                targetk = self.user_targetk_dict.get(u)
                itemk = self.item_containk_dict.get(i)
                if priork == None or targetk == None or itemk == None:
                    print(">> [Warning] Skip one data sample.")
                    continue
                user_priorks.append(priork)
                valid_priork_num.append(len(priork))
                user_targetks.append(targetk)
                valid_targetk_num.append(len(targetk))
                item_containks.append(itemk)
                valid_itemk_num.append(len(itemk))
            else:
                user_priorks.append([])
                valid_priork_num.append(0)
                user_targetks.append([])
                valid_targetk_num.append(0)
                item_containks.append([])
                valid_itemk_num.append(0)
        
        # pad 0 into the untidy list
        user_priorks = self.__padding_list(user_priorks)
        user_targetks = self.__padding_list(user_targetks)
        item_containks = self.__padding_list(item_containks)

        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        rating_tensor=torch.FloatTensor(ratings),
                                        priork_tensor=torch.LongTensor(user_priorks),
                                        targetk_tensor=torch.LongTensor(user_targetks),
                                        itemk_tensor=torch.LongTensor(item_containks),
                                        valid_priork_num=valid_priork_num,
                                        valid_targetk_num=valid_targetk_num,
                                        valid_itemk_num=valid_itemk_num)
        return DataLoader(dataset, batch_size = batch_size, shuffle=False, num_workers=0)

    def make_Y_train_loader(self, batch_size, enriched = True):
        users, items, ratings, user_priorks, user_targetks, item_containks = [], [], [], [], [],[]
        valid_priork_num, valid_targetk_num, valid_itemk_num = [],[],[] # 存储每一组data实际使用到的k数量
        for row in self.train_dataset:
            u = int(row[0])
            i = int(row[1])
            users.append(u)
            items.append(i)
            ratings.append(float(row[2]))
            if enriched:
                priork = self.user_priork_dict.get(u)
                targetk = self.user_targetk_dict.get(u)
                itemk = self.item_containk_dict.get(i)
                if priork == None or targetk == None or itemk == None:
                    print(">> [Warning] Skip one data sample.")
                    continue
                user_priorks.append(priork)
                valid_priork_num.append(len(priork))
                user_targetks.append(targetk)
                valid_targetk_num.append(len(targetk))
                item_containks.append(itemk)
                valid_itemk_num.append(len(itemk))
            else:
                user_priorks.append([])
                valid_priork_num.append(0)
                user_targetks.append([])
                valid_targetk_num.append(0)
                item_containks.append([])
                valid_itemk_num.append(0)
        
        # pad 0 into the untidy list
        user_priorks = self.__padding_list(user_priorks)
        user_targetks = self.__padding_list(user_targetks)
        item_containks = self.__padding_list(item_containks)

        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        rating_tensor=torch.FloatTensor(ratings),
                                        priork_tensor=torch.LongTensor(user_priorks),
                                        targetk_tensor=torch.LongTensor(user_targetks),
                                        itemk_tensor=torch.LongTensor(item_containks),
                                        valid_priork_num=valid_priork_num,
                                        valid_targetk_num=valid_targetk_num,
                                        valid_itemk_num=valid_itemk_num)
        return DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)

    ########################################### BPR Dataset Loader ##########################
    def make_BPR_valid_loader(self, batch_size):
        '''
        user,item,item: the pos and neg item here are the same
        the batch_size should be neg_test_sample_num + 1, defaultly 99
        '''

        users, items_pos, items_neg = [], [], []
        for row in self.validation_dataset:
            u = int(row[0])
            i = int(row[1])
            users.append(u)
            items_pos.append(i)
            items_neg.append(i)

        dataset = BPRDataset(user_tensor=torch.LongTensor(users),
                                item_pos_tensor=torch.LongTensor(items_pos),
                                item_neg_tensor=torch.LongTensor(items_neg),)
        return DataLoader(dataset, batch_size = batch_size, shuffle=False, num_workers=0)
    
    def make_BPR_test_loader(self, batch_size):
        '''
        user,item,item: the pos and neg item here are the same
        the batch_size should be neg_test_sample_num + 1, defaultly 99
        '''

        users, items_pos, items_neg = [], [], []
        for row in self.test_dataset:
            u = int(row[0])
            i = int(row[1])
            users.append(u)
            items_pos.append(i)
            items_neg.append(i)

        dataset = BPRDataset(user_tensor=torch.LongTensor(users),
                                item_pos_tensor=torch.LongTensor(items_pos),
                                item_neg_tensor=torch.LongTensor(items_neg),)
        return DataLoader(dataset, batch_size = batch_size, shuffle=False, num_workers=0)

    def make_BPR_train_loader(self, batch_size, n_neg_train):
        assert len(self.train_dataset)%(1+n_neg_train) == 0
        pos_num = int(len(self.train_dataset)/(1+n_neg_train))
        users, items_pos, items_neg = [], [], []
        
        for idx in range(pos_num):
            pos_idx = (1+n_neg_train)*idx
            pos_row = self.train_dataset[pos_idx]
            pos_u = int(pos_row[0])
            pos_i = int(pos_row[1])
            for n in range(1,n_neg_train+1):
                neg_idx = pos_idx+n
                neg_row = self.train_dataset[neg_idx]
                neg_u = int(neg_row[0])
                neg_i = int(neg_row[1])
                assert pos_u == neg_u
                users.append(pos_u)
                items_pos.append(pos_i)
                items_neg.append(neg_i)

        dataset = BPRDataset(user_tensor=torch.LongTensor(users),
                                item_pos_tensor=torch.LongTensor(items_pos),
                                item_neg_tensor=torch.LongTensor(items_neg),
                                        )
        return DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)

    ########################################### Knowledge Dataset Loader ##########################
    def make_K_test_loader(self, batch_size):
        # batch_size: a set of test samples
        fromks, toks, scores = [], [], []
        for row in self.klinking_test_dataset:
            fromk = int(row[0])
            tok = int(row[1])
            score = float(row[2])
            fromks.append(fromk)
            toks.append(tok)
            scores.append(score)

        dataset = KnowledgeLinkingDataset(fromk_tensor=torch.LongTensor(fromks),
                                        tok_tensor=torch.LongTensor(toks),
                                        score_tensor=torch.FloatTensor(scores))
        return DataLoader(dataset, batch_size = batch_size, shuffle=False, num_workers=0)

    def make_K_train_loader(self, batch_size):
        fromks, toks, scores = [], [], []
        for row in self.klinking_train_dataset:
            fromk = int(row[0])
            tok = int(row[1])
            score = float(row[2])
            fromks.append(fromk)
            toks.append(tok)
            scores.append(score)

        dataset = KnowledgeLinkingDataset(fromk_tensor=torch.LongTensor(fromks),
                                        tok_tensor=torch.LongTensor(toks),
                                        score_tensor=torch.FloatTensor(scores))
        return DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)