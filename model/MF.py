# Author: Hengchang Hu
# Date: 07 July, 202
# fork from https://github.com/guoyang9/NCF/edit/master/model.py
# Description: the pytorch implementation for NCF

import torch
import torch.nn as nn
import torch.nn.functional as F 

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class MF(nn.Module):
	def __init__(self, user_num, item_num, factor_num):    	
		super(MF, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors;
		"""		
		self.__alias__ = "MF"

		self.embed_user_GMF = nn.Embedding(user_num, factor_num)
		self.embed_item_GMF = nn.Embedding(item_num, factor_num)
		
		predict_size = factor_num 
		self.predict_layer = nn.Linear(predict_size, 1)

		self._init_weight_()

	def _init_weight_(self):
		""" We leave the weights initialization here. """
		nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
		nn.init.normal_(self.embed_item_GMF.weight, std=0.01)

		for m in self.modules():
			if isinstance(m, nn.Linear) and m.bias is not None:
				m.bias.data.zero_()
		

	def forward(self, user, item):
		embed_user_GMF = self.embed_user_GMF(user)
		embed_item_GMF = self.embed_item_GMF(item)
		output_GMF = embed_user_GMF * embed_item_GMF
		
		prediction = self.predict_layer(output_GMF)
		return prediction.view(-1)
	
	def get_alias(self):
		return self.__alias__