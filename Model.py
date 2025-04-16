import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
import math
from Utils.Utils import *

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class GCNModel(nn.Module):
	'''
		Multi-View Diffusion for multimodal recommender system.


	'''
	def __init__(self, image_embedding, text_embedding, audio_embedding=None, modal_fusion=True):
		super(GCNModel, self).__init__()
		
		self.sparse = True
		self.gcn_layer_num = 1
		self.edgeDropper = SpAdjDropEdge(args.keepRate)
		self.reg_weight = 1e-4
		# self.batch_size = 1024
		self.modal_fusion = modal_fusion

		# 新增可学习权重参数
		self.origin_weight = nn.Parameter(torch.ones(1))
		self.generation_weight = nn.Parameter(torch.ones(1))
		self.generation_weight2 = nn.Parameter(torch.ones(1))
		self.image_generation_weight = nn.Parameter(torch.ones(1))
		self.image_generation_weight = nn.Parameter(torch.ones(1))
		self.img_weight, self.txt_weight = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
		if audio_embedding is not None:
			self.aud_weight = nn.Parameter(torch.ones(1))

		# modal feature embedding
		if args.data == 'baby':
			self.image_embedding = denoise_norm(image_embedding, weight=args.image_norm)
			self.text_embedding = denoise_norm(text_embedding, weight=args.text_norm)
			self.audio_embedding = audio_embedding
			if audio_embedding is not None:
				self.audio_embedding = denoise_norm(audio_embedding, weight=args.audio_norm)
		else:
			# modal feature embedding
			self.image_embedding = image_embedding
			self.text_embedding = text_embedding
			self.audio_embedding = audio_embedding

		# user & item embdding
		self.user_embedding = nn.Embedding(args.user, args.latdim)    # self.user_embedding .shape: torch.Size([9308, 64]) self.item_id_embedding.shape: torch.Size([6710, 64])
		self.item_id_embedding = nn.Embedding(args.item, args.latdim)
		nn.init.xavier_uniform_(self.user_embedding.weight)
		nn.init.xavier_uniform_(self.item_id_embedding.weight)
		self.fusion_weight = nn.Parameter(torch.ones(3))
		self.res_scale = nn.Parameter(torch.ones(1))

		
		# 初始化权重参数
		nn.init.normal_(self.img_weight, mean=1.0, std=0.1)
		nn.init.normal_(self.txt_weight, mean=1.0, std=0.1)
		if audio_embedding is not None:
			nn.init.normal_(self.aud_weight, mean=1.0, std=0.1)

		# modal feature projection
		if self.image_embedding is not None:
			self.image_residual_project = nn.Sequential(
				nn.Linear(in_features=self.image_embedding.shape[1], out_features=args.latdim),
				nn.BatchNorm1d(args.latdim),
				nn.LeakyReLU(negative_slope=0.2),
				nn.Dropout(0.1)
			)
			self.image_modal_project = nn.Sequential(
				nn.Linear(in_features=args.latdim, out_features=args.latdim),
				nn.BatchNorm1d(args.latdim),
				nn.LeakyReLU(negative_slope=0.2),
				nn.Dropout(0.1)
			)

		if self.text_embedding is not None:
			self.text_residual_project = nn.Sequential(
				nn.Linear(in_features=self.text_embedding.shape[1], out_features=args.latdim),
				nn.BatchNorm1d(args.latdim),
				nn.LeakyReLU(negative_slope=0.2),
				nn.Dropout(0.1)
			)
			self.text_modal_project = nn.Sequential(
				nn.Linear(in_features=args.latdim, out_features=args.latdim),
				nn.BatchNorm1d(args.latdim),
				nn.LeakyReLU(negative_slope=0.2),
				nn.Dropout(0.1)
			)

		if self.audio_embedding is not None:
			self.audio_residual_project = nn.Sequential(
				nn.Linear(in_features=self.audio_embedding.shape[1], out_features=args.latdim),
				nn.BatchNorm1d(args.latdim),
				nn.LeakyReLU(negative_slope=0.2),
				nn.Dropout(0.1)
			)
			self.audio_modal_project = nn.Sequential(
				nn.Linear(in_features=args.latdim, out_features=args.latdim),
				nn.BatchNorm1d(args.latdim),
				nn.LeakyReLU(negative_slope=0.2),
				nn.Dropout(0.1)

			)

		self.softmax = nn.Softmax(dim=-1)

		self.gate_image_modal = nn.Sequential(
			nn.Linear(args.latdim, args.latdim),
			nn.BatchNorm1d(args.latdim),
			nn.Sigmoid()
		)

		self.gate_text_modal = nn.Sequential(
			nn.Linear(args.latdim, args.latdim),
			nn.BatchNorm1d(args.latdim),
			nn.Sigmoid()
		)

		self.gate_audio_modal = nn.Sequential(
			nn.Linear(args.latdim, args.latdim),
			nn.BatchNorm1d(args.latdim),
			nn.Sigmoid()
		)


		self.caculate_common = nn.Sequential(
		    nn.Linear(args.latdim, args.latdim),
			nn.BatchNorm1d(args.latdim),
		    nn.Tanh(),
		    nn.Linear(args.latdim, 1, bias=False)
		)
		
		# self.caculate_common = nn.Sequential(
		# 	nn.Linear(args.latdim, args.latdim),
		# 	nn.LeakyReLU(negative_slope=0.2),
		# 	nn.Linear(args.latdim, args.latdim//2),  # 增加隐藏层
		# 	nn.LayerNorm(args.latdim//2),
		# 	nn.Linear(args.latdim//2, 1, bias=False)
		# )
		self.init_modal_weight()

	def init_modal_weight(self):
		"""
		初始化模型权重
		"""
		# 对图像模态投影层中的线性层权重进行初始化（如果有图像模态嵌入的话）
		if self.image_embedding is not None:
			for layer in self.image_modal_project:
				if isinstance(layer, nn.Linear):
					nn.init.xavier_uniform_(layer.weight)
		# 对文本模态投影层中的线性层权重进行初始化（如果有文本模态嵌入的话）
		if self.text_embedding is not None:
			for layer in self.text_modal_project:
				if isinstance(layer, nn.Linear):
					nn.init.xavier_uniform_(layer.weight)
		# 对音频模态投影层中的线性层权重进行初始化（如果有音频模态嵌入的话）
		if self.audio_embedding is not None:
			for layer in self.audio_modal_project:
				if isinstance(layer, nn.Linear):
					nn.init.xavier_uniform_(layer.weight)
		
		# 对计算公共表示相关的线性层权重进行初始化
		for layer in self.caculate_common:
			if isinstance(layer, nn.Linear):
				nn.init.xavier_uniform_(layer.weight)
		
		# 对各模态的门控相关的线性层权重进行初始化
		for layer in self.gate_image_modal:
			if isinstance(layer, nn.Linear):
				nn.init.xavier_uniform_(layer.weight)
		for layer in self.gate_text_modal:
			if isinstance(layer, nn.Linear):
				nn.init.xavier_uniform_(layer.weight)
		for layer in self.gate_audio_modal:
			if isinstance(layer, nn.Linear):
				nn.init.xavier_uniform_(layer.weight)

		# 残差模块
		for layer in  self.image_residual_project:
				if isinstance(layer, nn.Linear):
					nn.init.xavier_uniform_(layer.weight)
		# 残差模块
		for layer in  self.text_residual_project:
				if isinstance(layer, nn.Linear):
					nn.init.xavier_uniform_(layer.weight)
		# 残差模块
		if self.audio_embedding is not None:
			for layer in  self.audio_residual_project:
					if isinstance(layer, nn.Linear):
						nn.init.xavier_uniform_(layer.weight)

		
	def getItemEmbeds(self):
		'''
			获取Item embedding
		'''
		return self.item_id_embedding.weight

	def getUserEmbeds(self):
		'''
			获取User embedding
		'''
		return self.user_embedding.weight
	
	def getImageFeats(self):
		'''
			获取图像模态特征
		'''
		if self.image_embedding is not None:
			x = self.image_residual_project(self.image_embedding)
			image_modal_feature = self.image_modal_project(x)
			image_modal_feature = self.res_scale * x + image_modal_feature
			# image_modal_feature = self.image_modal_projec(self.image_embedding)
		return image_modal_feature

	def getTextFeats(self):
		'''
			获取文本模态特征
		'''
		if self.text_embedding is not None:
			x = self.text_residual_project(self.text_embedding)
			text_modal_feature = self.text_modal_project(x)
			text_modal_feature = self.res_scale * x + text_modal_feature
		
			return text_modal_feature
	
	def getAudioFeats(self):
		'''
			获取音频模态特征
		'''
		if self.audio_embedding is not None:
			x = self.audio_residual_project(self.audio_embedding)
			audio_modal_feature = self.audio_modal_project(x)
			audio_modal_feature = self.res_scale * x + audio_modal_feature
		return audio_modal_feature
	
	def multimodal_feature_fusion_adj(self, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj):
		# 添加可学习的模态权重
		image_weight = torch.sigmoid(self.fusion_weight[0])
		text_weight = torch.sigmoid(self.fusion_weight[1])
		audio_weight = torch.sigmoid(self.fusion_weight[2])
		
		multimodal_feature_fusion_adj = (
			image_weight * diffusion_ii_image_adj +
			text_weight * diffusion_ii_text_adj +
			audio_weight * diffusion_ii_audio_adj
	)
	def user_item_GCN(self, adj):
		'''
			User-Item GCN
			original_ui_adj:size=(16018, 16018)
			diffusion_ui_adj:size=(6710, 6710)
			TODO: diffusion_ui_adj目前简单的进行融合

			original_ui_adj: tensor(indices=tensor([[    0, 10193, 10695,  ..., 16015, 16016, 16017],
								[    0,     0,     0,  ..., 16015, 16016, 16017]]),
				values=tensor([0.2500, 0.1443, 0.0606,  ..., 1.0000, 1.0000, 1.0000]),
				device='cuda:0', size=(16018, 16018), nnz=135100, layout=torch.sparse_coo) 
			
			diffusion_ui_adj: tensor(indices=tensor([[15828,     1,     2,  ..., 16012, 16013, 16016],
								[    0,     1,     2,  ..., 16012, 16013, 16016]]),
				values=tensor([0.0148, 1.0000, 1.0000,  ..., 2.0000, 2.0000, 2.0000]),
				device='cuda:0', size=(16018, 16018), nnz=51910, layout=torch.sparse_coo)

			adj: tensor(indices=tensor([[    0, 10193, 10695,  ..., 16012, 16013, 16016],
								[    0,     0,     0,  ..., 16012, 16013, 16016]]),
				values=tensor([0.2500, 0.1443, 0.0606,  ..., 2.0000, 2.0000, 2.0000]),
				device='cuda:0', size=(16018, 16018), nnz=187010, layout=torch.sparse_coo)
		'''
		#print("original_ui_adj:", original_ui_adj, "diffusion_ui_adj:", diffusion_ui_adj)
		# adj = original_ui_adj  #
		#print("adj:", adj)
		#adj = diffusion_ui_adj  # 
		#adj = original_ui_adj
		# adj1 = original_ui_adj
		# adj2 = diffusion_ui_adj 

		cat_embedding = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)

		all_embeddings = [cat_embedding]
		for i in range(self.gcn_layer_num):
			# temp_embeddings1 = torch.sparse.mm(adj1, cat_embedding)
			# cat_embedding = temp_embeddings1
			# all_embeddings += [cat_embedding]
			temp_embeddings2 = torch.sparse.mm(adj, cat_embedding)
			cat_embedding = temp_embeddings2
			all_embeddings += [cat_embedding]
		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
		content_embedding = all_embeddings
		#print("content_embedding:", content_embedding)

		return content_embedding


	def item_item_GCN(self,R, original_ui_adj, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj=None):
		'''
			Item-Item GCN
		'''
		# 获取 物品+ID 特征
		image_modal_feature = self.getImageFeats()
		image_item_id_embedding =  torch.multiply(self.item_id_embedding.weight, self.gate_image_modal(image_modal_feature))

		text_modal_feature = self.getTextFeats()
		text_item_id_embedding = torch.multiply(self.item_id_embedding.weight, self.gate_text_modal(text_modal_feature))

		if args.data == 'tiktok':
			audio_modal_feature = self.getAudioFeats()
			audio_item_id_embedding = torch.multiply(self.item_id_embedding.weight, self.gate_audio_modal(audio_modal_feature))

		# print("original_ui_adj.shape:", original_ui_adj.shape)
		# user-user adj
		self.R = R
		# print("original_ui_adj:", original_ui_adj)
		# print("R:", R)
		if self.sparse:
			for _ in range(self.gcn_layer_num):
				image_item_id_embedding = torch.sparse.mm(diffusion_ii_image_adj, image_item_id_embedding)
		else:
			for _ in range(self.gcn_layer_num):
				image_item_id_embedding = torch.mm(diffusion_ii_image_adj, image_item_id_embedding)

		image_user_embedding = torch.sparse.mm(self.R, image_item_id_embedding) 
		image_ui_embedding = torch.cat([image_user_embedding, image_item_id_embedding], dim=0)

		if self.sparse:
			for _ in range(self.gcn_layer_num):
				text_item_id_embedding = torch.sparse.mm(diffusion_ii_text_adj, text_item_id_embedding)
		else:
			for _ in range(self.gcn_layer_num):
				text_item_id_embedding = torch.mm(diffusion_ii_text_adj, text_item_id_embedding)
		text_user_embedding = torch.sparse.mm(self.R, text_item_id_embedding) 
		text_ui_embedding = torch.cat([text_user_embedding, text_item_id_embedding], dim=0)
		
		if args.data == 'tiktok':

					if self.sparse:

						for _ in range(self.gcn_layer_num):
							audio_item_id_embedding = torch.sparse.mm(diffusion_ii_audio_adj, audio_item_id_embedding)
					else:
						for _ in range(self.gcn_layer_num):
							audio_item_id_embedding = torch.mm(diffusion_ii_audio_adj, audio_item_id_embedding)

					audio_user_embedding = torch.sparse.mm(self.R, audio_item_id_embedding) 
					audio_ui_embedding = torch.cat([audio_user_embedding, audio_item_id_embedding], dim=0)


		return (image_ui_embedding, text_ui_embedding, audio_ui_embedding) if args.data == 'tiktok' else (image_ui_embedding, text_ui_embedding)


	def gate_attention_fusion(self, image_ui_embedding, text_ui_embedding, audio_ui_embedding=None):
		'''
			GAT Attention Fusion
		'''
		if args.data == 'tiktok':

			attention_common = torch.cat([self.caculate_common(image_ui_embedding), self.caculate_common(text_ui_embedding), self.caculate_common(audio_ui_embedding)], dim=-1)
			weight_common = self.softmax(attention_common)
			common_embedding = weight_common[:, 0].unsqueeze(dim=1) * image_ui_embedding + weight_common[:, 1].unsqueeze(dim=1) * text_ui_embedding + weight_common[:, 2].unsqueeze(dim=1) * audio_ui_embedding
			sepcial_image_ui_embedding = image_ui_embedding - common_embedding
			special_text_ui_embedding  = text_ui_embedding - common_embedding
			special_audio_ui_embedding = audio_ui_embedding - common_embedding

			return sepcial_image_ui_embedding, special_text_ui_embedding, special_audio_ui_embedding, common_embedding
		else:
			attention_common = torch.cat([self.caculate_common(image_ui_embedding), self.caculate_common(text_ui_embedding)], dim=-1)
			weight_common = self.softmax(attention_common)
			common_embedding = weight_common[:, 0].unsqueeze(dim=1) * image_ui_embedding + weight_common[:, 1].unsqueeze(dim=1) * text_ui_embedding 
			sepcial_image_ui_embedding = image_ui_embedding - common_embedding
			special_text_ui_embedding  = text_ui_embedding - common_embedding

			return sepcial_image_ui_embedding, special_text_ui_embedding, common_embedding


	def bpr_loss(self, anc_embeds, pos_embeds, neg_embeds):
		"""
		BPR loss计算函数:
		Args:
			anc_embeds: 用户嵌入向量，形状应为 [batch_size, embed_dim]
			pos_embeds: 正样本嵌入向量，形状应为 [batch_size, embed_dim]
			neg_embeds: 负样本嵌入向量，形状应为 [batch_size, embed_dim]
		Returns:
			bpr_loss: 计算得到的BPR损失值(一个标量)
		"""
		# 简单检查输入张量维度是否符合预期（这里只是简单示意，可根据实际情况完善）
		assert anc_embeds.dim() == 2, "用户嵌入向量维度应为2"
		assert pos_embeds.dim() == 2, "正样本嵌入向量维度应为2"
		assert neg_embeds.dim() == 2, "负样本嵌入向量维度应为2"
		assert anc_embeds.shape == pos_embeds.shape, "用户嵌入与正样本嵌入维度应匹配"
		assert anc_embeds.shape == neg_embeds.shape, "用户嵌入与负样本嵌入维度应匹配"

		# 计算正样本和负样本的得分：
		pos_scores = torch.sum(torch.mul(anc_embeds, pos_embeds), dim=-1)  # 计算用户与正样本物品的点积之和，作为正样本得分，形状为 [batch_size]
		neg_scores = torch.sum(torch.mul(anc_embeds, neg_embeds), dim=-1)  # 计算用户与负样本物品的点积之和，作为负样本得分，形状为 [batch_size]

		# 计算BPR损失
		diff_scores = pos_scores - neg_scores
		bpr_loss = -1 * torch.mean(F.logsigmoid(diff_scores))  # 计算正样本得分与负样本得分之差的logsigmoid，取反并计算平均值，促使正样本得分高于负样本得分，作为排序损失（一个数值）

		# 计算正则化损失
		regularizer = 1.0 / 2 * (anc_embeds ** 2).sum() + 1.0 / 2 * (pos_embeds ** 2).sum() + 1.0 / 2 * (neg_embeds ** 2).sum()
		regularizer = regularizer / args.batch
		emb_loss = self.reg_weight * regularizer 

		# 正则化损失2
		reg_loss = self.reg_loss() * args.reg
 
		return bpr_loss, emb_loss, reg_loss
	
	
	def infoNCE_loss(self, view1, view2,  temperature):
		'''
			InfoNCE loss
		'''
		view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
		pos_score = torch.sum((view1 * view2), dim=-1)
		pos_score = torch.exp(pos_score / temperature)

		neg_score = (view1 @ view2.T) / temperature
		neg_score = torch.exp(neg_score).sum(dim=1)
		contrast_loss = -1 * torch.log(pos_score / neg_score).mean()

		return contrast_loss

	def reg_loss(self):
		ret = 0
		ret += self.user_embedding.weight.norm(2).square()
		ret += self.item_id_embedding.weight.norm(2).square()
		return ret 

#	def forward(self, R, original_ui_adj, diffusion_ui_image_adj, diffusion_ui_text_adj, diffusion_ui_audio_adj, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj=None, diffusion_modal_fusion_ii_matrix=None):
	def forward(self, R, original_ui_adj, diffusion_ui_image_adj, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj=None, diffusion_modal_fusion_ii_matrix=None):
		'''
			GCN 前向过程:
				1. 多模态特征提取与fusion
				2. User-Item GCN
				3. Item-Item GCN
				4. GAT Attention Fusion
				5. Contrastive: BPR loss + InfoNCE
			
			Args:
				original_ui_adj: 原始的user-item graph
				diffusion_ui_adj: 扩散模型生成的user-item graph
				diffusion_ii_image_adj: 扩散模型生成的item-item image modal graph
				diffusion_ii_text_adj: 扩散模型生成的item-item  text modal graph
				diffusion_ii_audio_adj: 扩散模型生成的item-item audi modal graph

			Return:
				User Embedding, Item Embedding

				original_ui_adj: tensor(indices=tensor([[    0, 10193, 10695,  ..., 16015, 16016, 16017],
							[    0,     0,     0,  ..., 16015, 16016, 16017]]),
							values=tensor([0.2500, 0.1443, 0.0606,  ..., 1.0000, 1.0000, 1.0000]),
							device='cuda:0', size=(16018, 16018), nnz=135100, layout=torch.sparse_coo)

				diffusion_ui_adj: tensor(indices=tensor([[10272,     1,     2,  ..., 16012, 16013, 16016],
									[    0,     1,     2,  ..., 16012, 16013, 16016]]),
					values=tensor([0.0154, 1.0000, 1.0000,  ..., 2.0000, 2.0000, 2.0000]),
					device='cuda:0', size=(16018, 16018), nnz=51910, layout=torch.sparse_coo)
		'''
		# 多模态特征提取与fusion
		# print("original_ui_adj:", original_ui_adj)
		# print("diffusion_ui_adj:", diffusion_ui_adj)
		content_embedding1 = self.user_item_GCN(original_ui_adj)
		if args.data == 'baby':
			content_embedding = self.user_item_GCN(original_ui_adj + diffusion_ui_image_adj)
		
		else:
			content_embedding1 = self.user_item_GCN(original_ui_adj)
			content_embedding2 = self.user_item_GCN(diffusion_ui_image_adj)
			# content_embedding = self.user_item_GCN(original_ui_adj + diffusion_ui_image_adj)

			# content_embedding_image_modal = self.user_item_GCN(diffusion_ui_image_adj)
			#print("content_embedding_image_modal:", content_embedding_image_modal)
			# content_embedding_text_modal = self.user_item_GCN(diffusion_ui_text_adj)
			#content_embedding = (content_embedding_image_modal +  content_embedding_text_modal) / 
	
			weights = F.softmax(torch.stack([self.origin_weight, self.generation_weight]), dim=0)
			# print("weights:", weights)
			# if args.data == 'tiktok':
			# 	content_embedding_audio_modal = self.user_item_GCN(diffusion_ui_audio_adj)
			weight_1 = weights[0]
			weight_2 = weights[1]
			content_embedding = (
					weight_1 * content_embedding1 + 
					weight_2 * content_embedding2 
				)
		# content_embedding = content_embedding1
		#print("user-item gcn-------->content_embedding.shape", content_embedding.shape) # torch.Size([16018, 64])
			#content_embedding = (0.7 * content_embedding_image_modal + 0.2 *  content_embedding_text_modal +  0.1 * content_embedding_audio_modal) 
			# 使用softmax归一化权重
		# weights = F.softmax(torch.stack([self.img_weight, self.txt_weight, self.aud_weight]), dim=0)
		# content_embedding = (
		# 	weights[0] * content_embedding_image_modal + 
		# 	weights[1] * content_embedding_text_modal +
		# 	weights[2] * content_embedding_audio_modal
		# )
			
		if args.data == 'tiktok':
			
			if self.modal_fusion == True:
				diffusion_ii_image_adj += diffusion_modal_fusion_ii_matrix
				diffusion_ii_text_adj += diffusion_modal_fusion_ii_matrix
				diffusion_ii_audio_adj += diffusion_modal_fusion_ii_matrix

			# 物品-物品的GCN
			image_ui_embedding, text_ui_embedding, audio_ui_embedding = self.item_item_GCN(R, original_ui_adj, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj)
			######################################NORM#######################################
			#image_ui_embedding, text_ui_embedding, audio_ui_embedding = denoise_norm(image_ui_embedding), denoise_norm(text_ui_embedding), denoise_norm(audio_ui_embedding)

			sepcial_image_ui_embedding, special_text_ui_embedding, special_audio_ui_embedding, common_embedding = self.gate_attention_fusion(image_ui_embedding, text_ui_embedding, audio_ui_embedding)
			image_prefer_embedding = self.gate_image_modal(content_embedding) 
			text_prefer_embedding = self.gate_text_modal(content_embedding) 
			audio_prefer_embedding = self.gate_audio_modal(content_embedding) 

			sepcial_image_ui_embedding = torch.multiply(image_prefer_embedding, sepcial_image_ui_embedding)
			special_text_ui_embedding = torch.multiply(text_prefer_embedding, special_text_ui_embedding)
			special_audio_ui_embedding = torch.multiply(audio_prefer_embedding, special_audio_ui_embedding)
			weights = F.softmax(torch.stack([self.img_weight, self.txt_weight, self.aud_weight]), dim=0)
			side_embedding = ( sepcial_image_ui_embedding +  special_text_ui_embedding + special_audio_ui_embedding + common_embedding) / 4
			#side_embedding = (weights[0] * sepcial_image_ui_embedding + weights[1] * special_text_ui_embedding + weights[2] * special_audio_ui_embedding + common_embedding) / 4
			all_embedding = content_embedding + side_embedding
		else:
			if self.modal_fusion == True:
				diffusion_ii_image_adj += diffusion_modal_fusion_ii_matrix
				diffusion_ii_text_adj += diffusion_modal_fusion_ii_matrix


			image_ui_embedding, text_ui_embedding = self.item_item_GCN(R, original_ui_adj, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj=None)
			######################################NORM#######################################
			#image_ui_embedding, text_ui_embedding = denoise_norm(image_ui_embedding), denoise_norm(text_ui_embedding)

			sepcial_image_ui_embedding, special_text_ui_embedding, common_embedding = self.gate_attention_fusion(image_ui_embedding, text_ui_embedding, audio_ui_embedding=None)
			image_prefer_embedding = self.gate_image_modal(content_embedding) 
			text_prefer_embedding = self.gate_text_modal(content_embedding) 
			sepcial_image_ui_embedding = torch.multiply(image_prefer_embedding, sepcial_image_ui_embedding)
			special_text_ui_embedding = torch.multiply(text_prefer_embedding, special_text_ui_embedding)

			side_embedding = (sepcial_image_ui_embedding + special_text_ui_embedding + common_embedding) / 4
			# weights = F.softmax(torch.stack([self.img_weight, self.txt_weight]), dim=0)
			# side_embedding = (weights[0] * sepcial_image_ui_embedding + weights[1] *  special_text_ui_embedding + common_embedding) 
			all_embedding = content_embedding + side_embedding
		
		# split 
		all_embeddings_users, all_embeddings_items = torch.split(all_embedding, [args.user, args.item], dim=0)
		
		return all_embeddings_users, all_embeddings_items, side_embedding, content_embedding




def denoise_norm(emb1, weight=0.1):
	'''
		embedding denoise function
	'''
	# 核范数降噪
	# print("weight:", weight, "weight.item:", weight.item())
	# weight = weight.cuda()
	nuclear_norm_emb1= torch.linalg.svdvals(emb1).sum()

	# 可以根据需要调整核范数的权重
	# print("emb1.device:", emb1.device, "weight.device:", weight.device, "nuclear_norm_emb1:", nuclear_norm_emb1.device)
	emb1_norm = emb1 - weight * nuclear_norm_emb1

	return emb1_norm



class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return torch.spmm(adj, embeds)
	

class SpAdjDropEdge(nn.Module):
	def __init__(self, keepRate):
		super(SpAdjDropEdge, self).__init__()
		self.keepRate = keepRate

	def forward(self, adj):
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

		newVals = vals[mask] / self.keepRate
		newIdxs = idxs[:, mask]

		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
		

	

class FlipInterestDiffusion(nn.Module):
	'''
		实现稳定的兴趣扩散过程
		Args:
			steps: 扩散的时间步长
		Functions:
			-get_cum(): 获取时间步长的累积转移概率
			-q_sample():前向扩散过程
			-p_sample():反向扩散过程
			-training_losses():损失函数
	'''
	def __init__(self, steps=5, base_temp=1.0):
		# 建议添加稳定性参数
		super(FlipInterestDiffusion, self).__init__()  
		self.eps = 1e-8  # 缺失的稳定性参数
		self.steps = steps
		self.base_temp = base_temp  # 基础温度参数
		
		#self.alpha_bar0, self.alpha_bar1 = self.get_cum() # self.alpha_bar0：0->1累积转移概率，self.alpha_bar1 1->0累积转移概率
		'''
		self.alpha_bar0: tensor([0.0001, 0.0004, 0.0010, 0.0017, 0.0027])
		self.alpha_bar1: tensor([0.0010, 0.0018, 0.0023, 0.0026, 0.0027])

		gamma: tensor([1.0000e-04, 2.5750e-03, 5.0500e-03, 7.5250e-03, 1.0000e-02])
		epsilon: tensor([1.0000e-02, 7.5250e-03, 5.0500e-03, 2.5750e-03, 1.0000e-04])
		self.alpha_bar0: tensor([1.0000e-04, 2.5750e-07, 1.3004e-09, 9.7853e-12, 9.7853e-14])
		self.alpha_bar1: tensor([1.0000e-02, 7.5250e-05, 3.8001e-07, 9.7853e-10, 9.7853e-14])
		'''
		# print("self.alpha_bar0:", self.alpha_bar0)
		# print("self.alpha_bar1:", self.alpha_bar1)
		
	def _compute_sparsity(self, x):
		"""计算批次数据的稀疏性（0的比例）"""
		return (x == 0).float().mean()

	def _auto_schedule_params(self, x_start):
		"""根据输入数据自动调度 gamma 和 epsilon 参数"""
		sparsity = self._compute_sparsity(x_start)
		# Gamma (0->1 转移概率参数)
		# 稀疏性越高 -> gamma_start 越大（鼓励早期更多探索）
		gamma_start = 0.1 * (1 - sparsity) + 0.001  # 范围 [0.001, 0.1]
		gamma_end = gamma_start * 0.1  # 随 step 递减
		# Epsilon (1->0 转移概率参数)
		# 稀疏性越高 -> epsilon 越小（保护已有 1 的交互）
		epsilon_start = 0.005 * sparsity + 0.0001  # 范围 [0.0001, 0.005]
		epsilon_end = epsilon_start * 0.1

		return gamma_start, gamma_end, epsilon_start, epsilon_end


	def get_cum(self, x_start):
		"""动态生成基于数据的累积转移概率"""
		gamma_start, gamma_end, epsilon_start, epsilon_end = self._auto_schedule_params(x_start)
		# Gamma 调度（0->1 概率随时间递减）
		gamma = torch.linspace(gamma_start, gamma_end, self.steps)
		# Epsilon 调度（1->0 概率保持极低）
		epsilon = torch.linspace(epsilon_start, epsilon_end, self.steps)
		epsilon = torch.clamp(epsilon, max=0.01)  # 强制上限
		# 计算累积转移概率
		gamma_cum = 1 - torch.cumprod(1 - gamma, dim=0)
		epsilon_cum = 1 - torch.cumprod(1 - epsilon, dim=0)
		
		return gamma_cum.to(x_start.device), epsilon_cum.to(x_start.device)
	
	@staticmethod
	def generate_custom_noise(x, temp_scale=1.0, mode='randn'):
		"""带温度控制的自适应噪声生成"""
		if mode == 'randn':

			mean = x.float().mean()
			var = x.float().var(unbiased=True)
			std = torch.sqrt(var + 1e-8) * temp_scale  # 温度缩放标准差
			noise = torch.randn_like(x.float())
			return noise * std + mean
		if  mode == 'rand':
			noise = torch.rand_like(x.float())
			return noise

	
	def q_sample(self, x_start, t, temp_scale=1.0):
		"""前向扩散过程
			x_start:
			 		[tensor([[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					...,
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.]])

			x_start.shape: torch.Size([1024, 6710])
			t:时间步 t: tensor([1, 3, 1,  ..., 2, 1, 2], device='cuda:0') t.shape: torch.Size([1024]) 
			# alphas_cumprod: tensor([0.9999, 0.9997, 0.9995, 0.9992, 0.9990], device='cuda:0', dtype=torch.float64) 
			# alphas_cumprod.shape: torch.Size([5])

			self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) : tensor([[0.9997, 0.9997, 0.9997,  ..., 0.9997, 0.9997, 0.9997],
			[0.9992, 0.9992, 0.9992,  ..., 0.9992, 0.9992, 0.9992],
			[0.9997, 0.9997, 0.9997,  ..., 0.9997, 0.9997, 0.9997],
			...,
			[0.9995, 0.9995, 0.9995,  ..., 0.9995, 0.9995, 0.9995],
			[0.9997, 0.9997, 0.9997,  ..., 0.9997, 0.9997, 0.9997],
			[0.9995, 0.9995, 0.9995,  ..., 0.9995, 0.9995, 0.9995]],
			device='cuda:0')
			self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape).shape: torch.Size([1024, 6710])
		"""
		# 动态计算当前批次的参数
		gamma_cum, epsilon_cum = self.get_cum(x_start)
		
		# 提取当前时间步的累积概率
		self.alpha_bar0_t = self._extract_into_tensor(gamma_cum, t, x_start.shape)
		self.alpha_bar1_t = self._extract_into_tensor(epsilon_cum, t, x_start.shape)
		
		# 生成自适应噪声（带温度控制）
		noise = self.generate_custom_noise(x_start, temp_scale, mode = 'rand')
		# print("torch.sigmoid((self.alpha_bar0_t - noise) * self.base_temp):", torch.sigmoid((self.alpha_bar0_t - noise) * self.base_temp))
		# print("torch.sigmoid((self.alpha_bar1_t - noise) * self.base_temp) :", torch.sigmoid((self.alpha_bar1_t - noise) * self.base_temp) )
		# 计算软翻转概率
		flip_prob = torch.where(
			x_start == 0,
			torch.sigmoid((self.alpha_bar0_t - noise) * self.base_temp),  # 0->1
			torch.sigmoid((self.alpha_bar1_t - noise) * self.base_temp)   # 1->0
		)
		
		# 采样翻转
		flip_mask = torch.bernoulli(flip_prob)
		# print("flip_mask:", flip_mask)
		x_t = x_start.clone()
		x_t[flip_mask.bool()] = 1 - x_t[flip_mask.bool()]
		
		return x_t


	def p_sample(self, model, x_start, steps, bayesian_samplinge_schedule=True):
		'''
			model: 降噪模型
			x_start:
			 		[tensor([[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					...,
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.]])

			x_start.shape: torch.Size([1024, 6710])
			反向扩散过程 
				基于贝叶斯定理计算后验转移概率
				动态平衡探索与利用，保留可靠交互的同时探索潜在兴趣
				引入pos_weight缓解交互矩阵稀疏性问题


			x_start: tensor([[0., 0., 0.,  ..., 0., 0., 0.],
			[0., 0., 1.,  ..., 0., 0., 0.],
			[0., 0., 0.,  ..., 0., 0., 0.],
			...,
			[0., 0., 0.,  ..., 0., 0., 0.],
			[0., 0., 0.,  ..., 0., 0., 0.],
			[0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0')
			x_t: tensor([[1., 0., 1.,  ..., 1., 0., 0.],
					[1., 0., 1.,  ..., 1., 1., 1.],
					[1., 1., 1.,  ..., 1., 1., 1.],
					...,
					[1., 0., 1.,  ..., 0., 1., 0.],
					[1., 1., 1.,  ..., 0., 1., 1.],
					[0., 1., 1.,  ..., 1., 1., 1.]], device='cuda:0')
			prob: tensor([[0.7361, 0.6958, 0.7837,  ..., 0.7379, 0.7428, 0.7408],
					[0.7265, 0.7070, 0.7732,  ..., 0.7319, 0.7386, 0.7387],
					[0.7338, 0.7073, 0.7838,  ..., 0.7224, 0.7406, 0.7410],
					...,
					[0.7347, 0.7052, 0.7831,  ..., 0.7279, 0.7461, 0.7481],
					[0.7346, 0.6942, 0.7780,  ..., 0.7297, 0.7357, 0.7409],
					[0.7331, 0.6891, 0.7786,  ..., 0.7287, 0.7383, 0.7492]],
				device='cuda:0')
		'''
		batch_size = x_start.shape[0]
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps - 1] * batch_size).cuda() # t时间矩阵
			x_t = self.q_sample(x_start, t) # 每个batch内的张量都进行前向扩散
		# 反向过程迭代
		# print("self.steps:", self.steps)
		# [4, 3, 2, 1, 0]
		indices = list(range(self.steps))[::-1]

		for i in indices:
			#t = torch.tensor([i] * batch_size).cuda() # t时间矩阵 t: tensor([4, 4, 4,  ..., 4, 4, 4], device='cuda:0') t.shape: torch.Size([1024])
			# print("t:", t)
			# assert 0 <= i < self.steps, f'Invalid step index i={i} with total steps={self.steps}'
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			#assert t.max().item() < self.steps, f'Invalid t value {t.max().item()} at step {i}' # t时间矩阵 t: tensor([4, 4, 4,  ..., 4, 4, 4], device='cuda:0') t.shape: torch.Size([1024])
			logits, probs = self.p_interest_shift_probs(model, x_t, t) # torch.Size([1024, 6710])
			if bayesian_samplinge_schedule == True and i > 0:
				# 获取前向转移概率
				prev_alpha_bar0_t = self._extract_into_tensor(self.alpha_bar0_t, t-1, x_start.shape) # self.alpha_bar0[t-1]  torch.Size([1024, 6710])
				prev_alpha_bar1_t = self._extract_into_tensor(self.alpha_bar1_t, t-1, x_start.shape) # self.alpha_bar1[t-1]  torch.Size([1024, 6710])
				# 计算后验分布
				p0 = probs * (1 - prev_alpha_bar0_t) + (1 - probs) * prev_alpha_bar1_t
				p1 = probs * prev_alpha_bar0_t + (1 - probs) * (1 - prev_alpha_bar1_t)
				# 根据后验分布采样
				x_t = torch.bernoulli(p1 /(p0 + p1))
				#print("x_t:", x_t)

			else:
				x_t =  torch.bernoulli(probs)
	
		return x_t, probs

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats, text_feats, audio_feats):
		'''
			在扩散模型的损失函数设计中，需要考虑下面的问题:
			1. 传统的MSE损失要改为二元交叉熵损失
			2. 由于未交互的信息0值,远多余交互的信息1,因此需要考虑类别不均衡问题
			3. 
			Best epoch :  21  , Recall :  0.10848287112561175  , NDCG :  0.042286964164333356  , Precision 0.0054241435562805895
		'''
		# 动态类别权重计算
		pos_weight = torch.sum(1 - x_start) / (torch.sum(x_start) + 1e-8)
		# pos_weight =  (torch.sum(x_start) + 1e-8) / torch.sum(1 - x_start)
		batch_size = x_start.size(0)
		# 随机采样时间步
		t = torch.randint(0, self.steps, (batch_size,)).long().cuda() # t: tensor([1, 3, 1,  ..., 2, 1, 2], device='cuda:0') t.shape: torch.Size([1024]) 
		# 前向扩散
		x_t = self.q_sample(x_start, t) # torch.Size([1024, 6710])
		# 模型预测
		logits, probs = self.p_interest_shift_probs(model, x_t, t)
		###########################Focal Loss###########################
		gamma = 2.0  # 聚焦参数，越大对难样本关注越高2.0 
		alpha = 0.25 # 类别平衡基础系数0.25 
		# 计算概率并确保数值稳定
		p = torch.sigmoid(logits)
		p = torch.clamp(p, min=1e-7, max=1-1e-7)  # 防止log(0)
		# 动态调整alpha（结合原有pos_weight）
		adaptive_alpha = alpha * pos_weight.detach()  # 解耦梯度计算 0.0002
		# print("adaptive_alpha:", adaptive_alpha)
		# 计算正负样本mask
		pos_mask = x_start.float()
		neg_mask = 1 - pos_mask
		# 计算正负样本损失项
		pos_loss = -adaptive_alpha * (1 - p).pow(gamma) * pos_mask * torch.log(p)
		neg_loss = -(1 - adaptive_alpha) * p.pow(gamma) * neg_mask * torch.log(1 - p)
		
		# 合并损失并求均值
		#focal_loss = (pos_loss + neg_loss).mean()
		focal_loss = (pos_loss + neg_loss).sum() / (pos_mask.sum() + neg_mask.sum() + 1e-8)
		# 二元交叉熵损失
		bce_loss = F.binary_cross_entropy_with_logits(
			logits, x_start.float(), 
			pos_weight=pos_weight.cuda()  # 处理类别不平衡
		)
		# 计算对比损失
		# 原始图卷积的模态向量与生成的模态向量之间的对比损失，使得生成的模态正样本靠近原始
		gen_output, _ = self.p_sample(
			model=model,
			x_start=x_start,
			steps=self.steps,
			bayesian_samplinge_schedule=True
		)
		# print("gen_output:", gen_output)
		model_feat_embedding =  torch.multiply(itmEmbeds, model_feats)
		model_feat_embedding_origin = torch.mm(x_start, model_feat_embedding)
		model_feat_embedding_diffusion = torch.mm(gen_output, model_feat_embedding)
		cl_loss = self.infoNCE_loss(model_feat_embedding_origin, model_feat_embedding_diffusion, args.sparse_temp)

		text_model_feat_embedding =  torch.multiply(itmEmbeds, text_feats)
		text_model_feat_embedding_origin = torch.mm(x_start, text_model_feat_embedding)
		text_model_feat_embedding_diffusion = torch.mm(gen_output, text_model_feat_embedding)
		cl_loss_text = self.infoNCE_loss(text_model_feat_embedding_origin, text_model_feat_embedding_diffusion, args.sparse_temp)

		if args.data == 'tiktok':
			audio_model_feat_embedding =  torch.multiply(itmEmbeds, audio_feats)
			audio_model_feat_embedding_origin = torch.mm(x_start, audio_model_feat_embedding)
			audio_model_feat_embedding_diffusion = torch.mm(gen_output, audio_model_feat_embedding)
			cl_loss_audio = self.infoNCE_loss(audio_model_feat_embedding_origin, audio_model_feat_embedding_diffusion, args.sparse_temp)
			
		# KL散度项
		kl_loss = self._calc_kl_divergence(x_start, x_t, t, probs)
		# 课程学习权重
		curriculum_weight = torch.clamp(t.float() / self.steps, 0, 0.5) 
		kl_loss = (curriculum_weight * kl_loss).mean()
		if args.data == 'tiktok':
			total_loss = focal_loss +  kl_loss + args.ssl_gen1 * cl_loss   + args.ssl_gen2 * cl_loss_text + args.ssl_gen3 * cl_loss_audio 
		else:
			total_loss = focal_loss +  kl_loss + args.ssl_gen1 * cl_loss  + args.ssl_gen2 * cl_loss_text 
		#total_loss = bce_loss +  kl_loss + 0.01 * cl_loss
		#print("focal_loss:", total_loss, "bce_loss:", bce_loss, "kl_loss:", kl_loss, "cl_loss:", cl_loss)
		return total_loss

	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

	def _calc_kl_divergence(self, x0, xt, t, probs):
		"""稳健的KL散度计算"""
		# 真实后验概率
		post_probs = self._true_posterior(x0, xt, t).detach()
		
		# 数值稳定性处理
		post_probs = torch.clamp(post_probs, self.eps, 1-self.eps)
		probs = torch.clamp(probs.detach(), self.eps, 1-self.eps)
		
		kl = post_probs * (torch.log(post_probs) - torch.log(probs))
		kl += (1 - post_probs) * (torch.log(1 - post_probs) - torch.log(1 - probs))
		return kl.mean(dim=1)

	def _true_posterior(self, x0, xt, t):
		"""精确后验计算"""
		# alpha0 = self.alpha_bar0[t].view(-1,1)
		# alpha1 = self.alpha_bar1[t].view(-1,1)
		
		alpha0 = self._extract_into_tensor(self.alpha_bar0_t, t, x0.shape)
		alpha1 = self._extract_into_tensor(self.alpha_bar1_t, t, x0.shape)
		# 分子项
		case0 = (x0 == 0).float() * (1 - alpha0)
		case1 = (x0 == 1).float() * alpha1
		numerator = case0 + case1
		# 分母项
		denom0 = (x0 == 0).float() * (1 - alpha0 + alpha1)
		denom1 = (x0 == 1).float() * (alpha0 + 1 - alpha1)
		denominator = denom0 + denom1
		
		return numerator / (denominator + self.eps)
	
	def p_interest_shift_probs(self, model, x_t, t):
		logits = model(x_t, t) # model_output.shape: torch.Size([1024, 6710]
		# 计算各模态转移概率
		probs = torch.sigmoid(logits)
		return logits, probs

	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):

		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)
	
	def infoNCE_loss(self, view1, view2,  temperature):
		'''
			InfoNCE loss
		'''
		view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
		pos_score = torch.sum((view1 * view2), dim=-1)
		pos_score = torch.exp(pos_score / temperature)

		neg_score = (view1 @ view2.T) / temperature
		neg_score = torch.exp(neg_score).sum(dim=1)
		contrast_loss = -1 * torch.log(pos_score / neg_score).mean()

		return contrast_loss 
	


# Multi-Modal Fusion
class MultimodalFeatureFusion(nn.Module):
	'''
		多模态特征融合 (图像、文本、音频)
	'''
	def __init__(self, visual_feature_dim=128, text_feature_dim=768, audio_feature_dim=128, latdim=64, dropout_rate=0.5):
		'''
			Args:
				visual_feature_dim: 视觉特征维度
				text_feature_dim: 文本特征维度
				audio_feature_dim: 音频特征维度
				latdim: 隐藏层特征维度
		'''
		super(MultimodalFeatureFusion, self).__init__()
		self.latdim = latdim
		self.dropout_rate = dropout_rate
		self.modal_num = 3 if audio_feature_dim > 0 else 2

		self.visual_encoder = self._build_encoder(visual_feature_dim)
		self.text_encoder = self._build_encoder(text_feature_dim)
		self.audio_encoder = self._build_encoder(audio_feature_dim)

		self.attention_project = nn.Linear(
			in_features=self.latdim * self.modal_num, 
			out_features=self.modal_num
		)

	def _build_encoder(self, feature_dim):
		if feature_dim:
			return nn.Sequential(
				nn.Linear(feature_dim, self.latdim),
				nn.BatchNorm1d(self.latdim),
				nn.LeakyReLU(negative_slope=0.2),
				nn.Dropout(self.dropout_rate)
			)
		return None

	def forward(self, visual_feature, text_feature, audio_feature):
		'''
			@Desc: 前向阶段
			Args:
				visual_feature: 视觉特征
				text_feature:  文本特征
				audio_feature: 音频特征
			Return:
				multimodal_fusion: 多模态融合特征
		'''
		embeddings = []
		if visual_feature is not None:
			visual_embedding = self.visual_encoder(visual_feature)
			embeddings.append(visual_embedding)
		if text_feature is not None:
			text_embedding = self.text_encoder(text_feature)
			embeddings.append(text_embedding)
		if audio_feature is not None:
			audio_embedding = self.audio_encoder(audio_feature)
			embeddings.append(audio_embedding)

		concat_embedding = torch.cat(embeddings, dim=1)
		attention = torch.softmax(self.attention_project(concat_embedding), dim=1)

		multimodal_fusion = 0
		if visual_feature is not None:
			multimodal_fusion += attention[:, 0].unsqueeze(1) * visual_embedding
		if text_feature is not None:
			multimodal_fusion += attention[:, 1].unsqueeze(1) * text_embedding
		if audio_feature is not None:
			multimodal_fusion += attention[:, 2].unsqueeze(1) * audio_embedding

		return multimodal_fusion
	
	
import torch
import torch.nn as nn
import math


class ModalDenoiseTransformer(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.2):
		super().__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.time_emb_dim = emb_size
		
		# 时间嵌入层
		self.time_emb = nn.Sequential(
			nn.Linear(emb_size, 4*emb_size),
			nn.SiLU(),
			nn.Linear(4*emb_size, emb_size)
		)
		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
		# 输入投影层
		self.input_proj = nn.Linear(in_dims + emb_size, dim_feedforward)
		
		# Transformer Decoder
		decoder_layer = nn.TransformerDecoderLayer(
			d_model=dim_feedforward,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True)
		self.transformer_decoder = nn.TransformerDecoder(
			decoder_layer, 
			num_layers=num_layers)
		# 输出投影
		self.output_proj = nn.Sequential(
			nn.Linear(dim_feedforward, dim_feedforward//2),
			nn.LayerNorm(dim_feedforward//2),
			nn.GELU(),
			nn.Linear(dim_feedforward//2, out_dims))
		# AdaLN（自适应层归一化）
		self.adaLN_modulation = nn.Sequential(
			nn.SiLU(),
			nn.Linear(emb_size, 2*dim_feedforward))
		# 初始化
		self.apply(self._init_weights)
	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			nn.init.xavier_uniform_(module.weight)
			if module.bias is not None:
				nn.init.constant_(module.bias, 0.01)
				
	def forward(self, x, timesteps):
		"""
		Args:
			x: [batch_size, in_dims]
			timesteps: [batch_size]
			x: 
			tensor([[1., 0., 0.,  ..., 1., 0., 0.],
			[0., 0., 0.,  ..., 1., 0., 1.],
			[1., 1., 1.,  ..., 1., 1., 1.],
			...,
			[0., 1., 1.,  ..., 0., 0., 0.],
			[0., 0., 1.,  ..., 1., 0., 0.],
			[0., 1., 0.,  ..., 1., 0., 0.]], device='cuda:0')
			time_emb: 
			tensor([[-0.9900,  0.8891,  0.9972,  ...,  0.0753,  0.0119,  0.0019],
					[-0.9900,  0.8891,  0.9972,  ...,  0.0753,  0.0119,  0.0019],
					[-0.9900,  0.8891,  0.9972,  ...,  0.0753,  0.0119,  0.0019],
					...,
					[-0.9900,  0.8891,  0.9972,  ...,  0.0753,  0.0119,  0.0019],
					[-0.9900,  0.8891,  0.9972,  ...,  0.0753,  0.0119,  0.0019],
					[-0.9900,  0.8891,  0.9972,  ...,  0.0753,  0.0119,  0.0019]],
				device='cuda:0')
		Returns:
			out: [batch_size, out_dims]
		time_emb.shape: torch.Size([1024, 10])
		x.shape: torch.Size([1024, 6710])
		"""
		#print("timesteps:", timesteps) # tensor([3, 4, 3,  ..., 2, 0, 1], device='cuda:0')
		# 生成时间嵌入
		# time_emb = self.get_timestep_embedding(timesteps, self.time_emb_dim)  # [B, emb_size]
		# time_emb = self.time_emb(time_emb)  # [B, emb_size]
		# print("timesteps.device:", timesteps.device)
		# freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		#print("time_emb.shape:", time_emb.shape)
		# print("time_emb.device:", time_emb.device)
		# print("time_emb:", time_emb)
		time_emb = self.emb_layer(time_emb.cuda())
		# print("x.shape:", x.shape)
		# print("time_emb.shape:", time_emb.shape)
		# print("x:", x)
		# print("time_emb:", time_emb)
		# 输入投影
		h = torch.cat([x, time_emb], dim=-1)  # [B, in_dims+emb_size]
		h = self.input_proj(h)  # [B, dim_feedforward]
		
		# 添加序列维度（模拟单步解码）
		h = h.unsqueeze(1)  # [B, 1, dim_feedforward]
		
		# AdaLN调制
		shift, scale = self.adaLN_modulation(time_emb).chunk(2, dim=1)
		h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
		
		# Transformer解码过程
		memory = torch.zeros_like(h)  # 空memory（自回归模式）
		out = self.transformer_decoder(tgt=h, memory=memory)  # [B, 1, dim_feedforward]
		
		# 输出投影
		out = out.squeeze(1)  # [B, dim_feedforward]
		out = self.output_proj(out)  # [B, out_dims]
		
		return out

	# def get_timestep_embedding(self, timesteps, time_emb_dim):
	# 	"""
	# 	生成符合Transformer标准的位置编码
	# 	Args:
	# 		timesteps: [batch_size]
	# 		time_emb_dim: 嵌入维度
	# 	Returns:
	# 		emb: [batch_size, time_emb_dim]
	# 	"""
	# 	#print("timesteps:", timesteps) # timesteps: tensor([3, 4, 3,  ..., 2, 0, 1], device='cuda:0')
	# 	half_dim = time_emb_dim // 2
	# 	# === 修正1: 正确的频率系数计算 ===
	# 	scale = math.log(10000) / (half_dim)  # 分母使用half_dim
	# 	div_term = torch.exp(torch.arange(half_dim) * -scale)
		
	# 	# === 修正2: 显式广播计算 ===
	# 	angles = timesteps[:, None] * div_term[None, :]  # [batch_size, half_dim]
		
	# 	# 生成正弦和余弦嵌入
	# 	emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [batch_size, time_emb_dim]
		
	# 	# 处理奇数维度
	# 	if time_emb_dim % 2 == 1:
	# 		emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
	# 	return emb