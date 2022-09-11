# Author: Hengchang Hu
# Date: 24 Jan, 2021
# Description: used for the latent factor learning for knowledge unit

import torch
import torch.nn as nn
import torch.nn.functional as F 

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class KnowledgeMF(nn.Module):
	def __init__(self, knowledge_num, factor_num): 
		super(KnowledgeMF, self).__init__()
		"""
		user_num: number of knowledge;
		factor_num: number of predictive factors;
		"""
		self.__alias__ = "KnowledgeMF"

		self.embed_k_GMF = nn.Embedding(knowledge_num, factor_num)
		
		predict_size = factor_num 
		self.predict_layer = nn.Linear(predict_size, 1)

		self._init_weight_()

	def _init_weight_(self):
		""" We leave the weights initialization here. """
		nn.init.normal_(self.embed_k_GMF.weight, std=0.01)

		for m in self.modules():
			if isinstance(m, nn.Linear) and m.bias is not None:
				m.bias.data.zero_()
		

	def forward(self, fromk, tok):
		embed_fromk_GMF = self.embed_k_GMF(fromk)
		embed_tok_GMF = self.embed_k_GMF(tok)
		output_GMF = embed_fromk_GMF * embed_tok_GMF
		
		prediction = self.predict_layer(output_GMF)
		return prediction.view(-1)
	
	def get_alias(self):
		return self.__alias__