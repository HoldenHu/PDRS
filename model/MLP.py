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

class MLP(nn.Module):
	def __init__(self, user_num, item_num, factor_num, num_layers, dropout):    	
		super(MLP, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors;
		num_layers: the number of layers in MLP model;
		dropout: dropout rate between fully connected layers;
		"""		
		self.__alias__ = "MLP"
		self.dropout = dropout

		self.embed_user_MLP = nn.Embedding(
				user_num, factor_num * (2 ** (num_layers - 1)))
		self.embed_item_MLP = nn.Embedding(
				item_num, factor_num * (2 ** (num_layers - 1)))

		MLP_modules = []
		for i in range(num_layers):
			input_size = factor_num * (2 ** (num_layers - i))
			MLP_modules.append(nn.Dropout(p=self.dropout))
			MLP_modules.append(nn.Linear(input_size, input_size//2))
			MLP_modules.append(nn.ReLU())
		self.MLP_layers = nn.Sequential(*MLP_modules)

		predict_size = factor_num
		self.predict_layer = nn.Linear(predict_size, 1)

		self._init_weight_()

	def _init_weight_(self):
		""" We leave the weights initialization here. """
		nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
		nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

		for m in self.MLP_layers:
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
		nn.init.kaiming_uniform_(self.predict_layer.weight, 
								a=1, nonlinearity='sigmoid')

		for m in self.modules():
			if isinstance(m, nn.Linear) and m.bias is not None:
				m.bias.data.zero_()
		

	def forward(self, user, item):
		embed_user_MLP = self.embed_user_MLP(user)
		embed_item_MLP = self.embed_item_MLP(item)

		interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
		output_MLP = self.MLP_layers(interaction)

		prediction = self.predict_layer(output_MLP)
		return prediction.view(-1)

	def predict(self, user, item):
		# return the score, inputs and outputs are numpy arrays
		# for key in feed_dict:
		# 	if type(feed_dict[key]) != type(None):
		# 		_tensor = torch.from_numpy(feed_dict[key])
		# 		_tensor_to_gpu = _tensor.to(dtype = torch.long, device = device)
		# 		feed_dict[key] = _tensor_to_gpu
		# output_scores = self.forward(feed_dict)
		return False
	
	def get_alias(self):
		return self.__alias__