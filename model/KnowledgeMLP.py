# Author: Hengchang Hu
# Date: 07 July, 2020
# fork from https://github.com/guoyang9/NCF/edit/master/model.py
# Description: the pytorch implementation for NCF

import torch
import torch.nn as nn
import torch.nn.functional as F 

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class KnowledgeMLP(nn.Module):
	def __init__(self, knowledge_num, factor_num, num_layers, dropout):    	
		super(KnowledgeMLP, self).__init__()
		"""
		> used for get the embedding of each knowledge concept
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors;
		num_layers: the number of layers in MLP model;
		dropout: dropout rate between fully connected layers;
		"""		
		self.__alias__ = "KnowledgeMLP"
		self.dropout = dropout
		self.logistic = nn.Sigmoid()

		self.embed_k_MLP = nn.Embedding(
				knowledge_num, factor_num * (2 ** (num_layers - 2)))

		MLP_modules = []
		for i in range(1, num_layers):
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
		nn.init.normal_(self.embed_k_MLP.weight, std=0.01)

		for m in self.MLP_layers:
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
		nn.init.kaiming_uniform_(self.predict_layer.weight, 
								a=1, nonlinearity='sigmoid')

		for m in self.modules():
			if isinstance(m, nn.Linear) and m.bias is not None:
				m.bias.data.zero_()
		

	def forward(self, fromk, tok):
		embed_fromk_MLP = self.embed_k_MLP(fromk)
		embed_tok_MLP = self.embed_k_MLP(tok)

		interaction = torch.cat((embed_fromk_MLP, embed_tok_MLP), -1)
		output_MLP = self.MLP_layers(interaction)

		prediction = self.predict_layer(output_MLP)
		prediction = self.logistic(prediction)
		return prediction.view(-1)
	
	def get_alias(self):
		return self.__alias__