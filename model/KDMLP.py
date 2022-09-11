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

class KDMLP(nn.Module):
	def __init__(self, user_num, item_num, knowledge_num, factor_num, kfactor_num, num_layers, dropout, use_priork=True, use_targetk=True, use_itemk=True):    	
		super(KDMLP, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		knowledge num: number of knowledge;
		factor_num: number of predictive factors;
		num_layers: the number of layers in MLP model, at least should be 2;
		dropout: dropout rate between fully connected layers;
		----------
		KDMLP and KnowledgeMLP share the embed_k_MLP parameters when training
		input: (embed_user, embed_item, embed_priork, embed_targetk, embed_itemk)
		output: (factor_num)
		"""		
		self.__alias__ = "KDMLP"
		self.dropout = dropout
		self.knowledge_input_count = 0
		if use_priork == True: self.knowledge_input_count += 1 
		if use_targetk == True: self.knowledge_input_count += 1 
		if use_itemk == True: self.knowledge_input_count += 1 
		self.use_priork, self.use_targetk, self.use_itemk = use_priork, use_targetk, use_itemk

		self.embed_user_MLP = nn.Embedding(
				user_num, factor_num * (2 ** (num_layers - 2)))
		self.embed_item_MLP = nn.Embedding(
				item_num, factor_num * (2 ** (num_layers - 2)))
		self.embed_k_MLP = nn.Embedding(
				knowledge_num, kfactor_num * (2 ** (num_layers - 2)))
		
		MLP_modules = []
		for i in range(num_layers):
    		# 3: 8 -> 4 -> 2 -> 1 => 10 -> 4 -> 2 -> 1
			# input_len: 4           input_len: 2
			# 4: 16 -> 8 -> 4 ->2 -> 1 => 20 -> 8 -> 4 -> 2 -> 1
			# input_len: 8           input_len: 4
			if i == 0:
				input_size = factor_num * 2 * (2 ** (num_layers - 2)) + kfactor_num * self.knowledge_input_count * (2 ** (num_layers - 2))
				MLP_modules.append(nn.Dropout(p=self.dropout))
				MLP_modules.append(nn.Linear(input_size, factor_num * (2 ** (num_layers - 1))))
				MLP_modules.append(nn.ReLU())
			else:
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
		nn.init.normal_(self.embed_k_MLP.weight, std=0.01)

		for m in self.MLP_layers:
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
		nn.init.kaiming_uniform_(self.predict_layer.weight, 
								a=1, nonlinearity='sigmoid')

		for m in self.modules():
			if isinstance(m, nn.Linear) and m.bias is not None:
				m.bias.data.zero_()
		
	def __flatten_knowledge_embedding(self, knowledge_tensor, valid_num, embed_k_MLP):
		'''
		knowledge_tensor = Tensor([[0,1,2],[1,2,3],[2,3,4]]), device('cpu')
		valid_num = [0,1,2], device('cpu)
		embed_k_MLP = self.embed_k_MLP, device('cuda')
		Return:
			sample number(3 here) * lalent factor
		'''
		embed_k_MLP = embed_k_MLP.to('cpu')
		embed_k = embed_k_MLP(knowledge_tensor) # sample number*max_len*latent factor
		embed_k.detach_()
		embed_k.requires_grad = False

		mask = torch.zeros(embed_k.shape)
		for i in range(embed_k.shape[0]):
			mask[i,torch.arange(valid_num[i])] = 1
		'''
		mask:
		tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

        [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

        [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])
		'''
		k = torch.sum(mask * embed_k, dim=1, keepdim=False)
		div = valid_num.type(torch.FloatTensor).repeat([k.shape[-1],1]).T
		output_k = k / div
		output_k[output_k != output_k] = 0 # set all nan value to 0
		return output_k.to('cuda') # sample_num * latent_factor


	def forward(self, user, item, priork_tensor, targetk_tensor, itemk_tensor, valid_priork_num, valid_targetk_num, valid_itemk_num):
		'''
		user, item: device('cuda:0')
		priork_tensor, targetk_tensor, itemk_tensor, valid_priork_num, valid_targetk_num, valid_itemk_num: device('cpu')
		'''
		embed_user_MLP = self.embed_user_MLP(user)
		embed_item_MLP = self.embed_item_MLP(item)
		if self.knowledge_input_count == 0: # pure rs
			interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
		elif self.knowledge_input_count == 1: 
			embed_k_input = None
			if self.use_priork:
				embed_priork_MLP = self.__flatten_knowledge_embedding(priork_tensor,valid_priork_num, self.embed_k_MLP)
				embed_k_input = embed_priork_MLP
			elif self.use_targetk:
				embed_targetk_MLP = self.__flatten_knowledge_embedding(targetk_tensor,valid_targetk_num, self.embed_k_MLP)
				embed_k_input = embed_targetk_MLP
			elif self.use_itemk:
				embed_itemk_MLP = self.__flatten_knowledge_embedding(itemk_tensor,valid_itemk_num, self.embed_k_MLP)
				embed_k_input = embed_itemk_MLP
			interaction = torch.cat((embed_user_MLP, embed_item_MLP, embed_k_input), -1)
		elif self.knowledge_input_count == 2:
			embed_k_input1,embed_k_input2 = None,None
			if not self.use_priork:
				embed_targetk_MLP = self.__flatten_knowledge_embedding(targetk_tensor,valid_targetk_num, self.embed_k_MLP)
				embed_k_input1 = embed_targetk_MLP
				embed_itemk_MLP = self.__flatten_knowledge_embedding(itemk_tensor,valid_itemk_num, self.embed_k_MLP)
				embed_k_input2 = embed_itemk_MLP
			elif not self.use_targetk:
				embed_priork_MLP = self.__flatten_knowledge_embedding(priork_tensor,valid_priork_num, self.embed_k_MLP)
				embed_k_input1 = embed_priork_MLP
				embed_itemk_MLP = self.__flatten_knowledge_embedding(itemk_tensor,valid_itemk_num, self.embed_k_MLP)
				embed_k_input2 = embed_itemk_MLP
			elif not self.use_itemk:
				embed_priork_MLP = self.__flatten_knowledge_embedding(priork_tensor,valid_priork_num, self.embed_k_MLP)
				embed_k_input1 = embed_priork_MLP
				embed_targetk_MLP = self.__flatten_knowledge_embedding(targetk_tensor,valid_targetk_num, self.embed_k_MLP)
				embed_k_input2 = embed_targetk_MLP
			interaction = torch.cat((embed_user_MLP, embed_item_MLP, embed_k_input1, embed_k_input2), -1)
		elif self.knowledge_input_count == 3: # defaultly input all knowledge information
			embed_priork_MLP = self.__flatten_knowledge_embedding(priork_tensor,valid_priork_num, self.embed_k_MLP)
			embed_targetk_MLP = self.__flatten_knowledge_embedding(targetk_tensor,valid_targetk_num, self.embed_k_MLP)
			embed_itemk_MLP = self.__flatten_knowledge_embedding(itemk_tensor,valid_itemk_num, self.embed_k_MLP)
			interaction = torch.cat((embed_user_MLP, embed_item_MLP, embed_priork_MLP, embed_targetk_MLP, embed_itemk_MLP), -1)

		output_MLP = self.MLP_layers(interaction)

		prediction = self.predict_layer(output_MLP)
		return prediction.view(-1)
	
	def get_alias(self):
		return self.__alias__