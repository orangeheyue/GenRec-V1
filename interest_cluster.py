# Desc: Clustering for the multimodal features.
# Author: GenRec-V1
# Update: 2025/3/31

import numpy as np
import math 
import torch
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# MultimodalCluster
class MultimodalCluster(object):
	'''
		Desc:多模态特征聚类
			- Modal-Specific Clustering, 多模态专属聚类
			- Modal Fusion Clustering, 多模态融合聚类
		Args:

		Attention: The search space is in this:
		###################################FIND The Cluster Range###################################
		# Baby: interest_min: 3 interest_mean: 7 interest_max: 100
		# TikTok: interest_min: 0 interest_mean: 7 interest_max: 603
		# Sport: interest_min: 3 interest_mean: 7 interest_max: 237
		###################################FIND The Cluster Range###################################
	'''
	def __init__(self
			  ,num_cluster_visual_modal
			  ,num_cluster_text_modal
			  ,num_cluster_audio_modal
			  ,num_cluster_fusion_modal
			  ,kmeans_cluster_num
			  ,spectral_cluster_num
			  ,sim_top_k
			  ,use_auto_optimal_k
			  ,kmeans_cluster_num_min
			  ,kmeans_cluster_num_mean
			  ,kmeans_cluster_num_max
			  ,kmeans_stride
		):
		super(MultimodalCluster, self).__init__()
		self.num_cluster_visual_modal = num_cluster_visual_modal
		self.num_cluster_text_modal =  num_cluster_text_modal
		self.num_cluster_audio_modal = num_cluster_audio_modal
		self.num_cluster_fusion_modal = num_cluster_fusion_modal 
		self.kmeans_cluster_num =  kmeans_cluster_num
		self.spectral_cluster_num = spectral_cluster_num
		self.sim_top_k = sim_top_k
		self.use_auto_optimal_k = use_auto_optimal_k
		self.stand_norm = StandardScaler()
		self.kmeans_cluster_num_min = kmeans_cluster_num_min
		self.kmeans_cluster_num_mean = kmeans_cluster_num_mean
		self.kmeans_cluster_num_max = kmeans_cluster_num_max
		self.stride = kmeans_stride

	def multimodal_specific_cluster(self, specific_modal_features, modality, optimial_cluster_num):
		'''
			Desc:对某种模态特征进行聚类,基于Kmeans聚类方法
			Args:
				specific_modal_features: 模态特征,若有1024个物品,图像特征维度为64,则形状为 (1024, 64)
			Return:
				items_cluster_labels(每个物品id被聚类成的物品类别标签), (1024,), 保存的是0-聚类数的整数列表
				for example:  
				if cluster number is 3, then labels = np.array([0, 2, 1, 0, 1])
		'''
		print("self.use_auto_optimal_k:", self.use_auto_optimal_k)
		# print("specific_modal_features:", specific_modal_features)
		if isinstance(specific_modal_features, torch.Tensor):
			specific_modal_features = specific_modal_features.cpu().detach().numpy()
		specific_modal_feature_norm = self.stand_norm.fit_transform(specific_modal_features)
		best_kmeans_cluster_num = self.get_kmeans_cluster_optimal_num(specific_modal_feature_norm) if self.use_auto_optimal_k else optimial_cluster_num
		print("modality:", modality, "best_kmeans_cluster_num:", best_kmeans_cluster_num)
		kmeans_cluster = KMeans(n_clusters=best_kmeans_cluster_num).fit(specific_modal_feature_norm)
		items_cluster_labels = kmeans_cluster.labels_
		return items_cluster_labels
	
	def get_kmeans_cluster_optimal_num(self, stand_feature_matrix):
		'''
			Desc: Automatically determine the optimal number of clusters k for K-means clustering.
			Args:
				stand_feature_matrix, shape= (6710, 128) 
			Return:
				最佳聚类数, Int类型 
			Attention:计算量可能比较大，只需要运行一次就可以提前获取到最优的聚类数量
		'''
		print("--------------->auto get_kmeans_cluster_optimal_num:")
		# print("stand_feature_matrix:", stand_feature_matrix)
		distortions = [] # 存储每个k值对应的簇内平方和. 簇内平方和:所有样本到其所属簇中心的距离平方和
		# k_range_max = stand_feature_matrix.shape[0] # 物品的个数, 6710
		# print("k_range_max:", k_range_max)
		for i in range(self.kmeans_cluster_num_min, self.kmeans_cluster_num_max, self.stride):
			kmeans_cluster = KMeans(n_clusters=i).fit(stand_feature_matrix)
			distortions.append(kmeans_cluster.inertia_) # 
		# 计算二阶差分 （寻找Inertia下降速度的突变点（曲率最大点））
		print("distortions:", distortions)
		diff2 = np.diff(np.diff(distortions))
		#  二阶差分最大值的索引 + min + 1
		print("np.argmin(diff2):", np.argmin(diff2))
		best_cluster_nums = np.argmin(diff2) + self.kmeans_cluster_num_min + 1 
		return best_cluster_nums

	def multimodal_fusion_cluster(self, fusion_modal_feature):
		'''
			Desc: 对多模态融合的特征进行聚类,基于频谱聚类方法
			Args:
				fusion_modal_feature: 多模态融合特征,经过多模态Fusion模块后的特征,形状仍为(1024, 64)
			Return:items_cluster_labels(每个物品id被聚类成的物品类别标签), (1024,), 保存的是0-聚类数的整数列表 (每个用户喜欢的物品类别)
		'''
		# 构建相似矩阵
		sim_feature_matrix = cosine_similarity(fusion_modal_feature)
		# 相似矩阵稀疏化,目的是只保留最相似的topk
		item_size = fusion_modal_feature.shape[0]
		for i in range(item_size):
			idx = np.argpartition(sim_feature_matrix[i], -self.sim_top_k)[-self.sim_top_k:]
			bool_mask = np.ones(item_size, bool)
			bool_mask[idx] = False
			sim_feature_matrix[i, bool_mask] = 0 
		sim_feature_matrix_sparse =  np.maximum(sim_feature_matrix, sim_feature_matrix.T)
		best_spectral_num_cluster = self.get_spectral_cluster_optimal_num(sim_feature_matrix_sparse) if self.use_auto_optimal_k else self.spectral_cluster_num
		spectral_cluster =  SpectralClustering(n_clusters=best_spectral_num_cluster, affinity='precomputed')
		items_cluster_labels = spectral_cluster.fit_predict(sim_feature_matrix_sparse)
		return items_cluster_labels
	
	def get_spectral_cluster_optimal_num(self, sim_feature_matrix):
		'''
			Desc: Automatically determine the optimal number of clusters k for spectral clustering
			获取谱聚类最优聚类数(使用谱间隙法Spectral Gap确定聚类数,本质是找到有显著的差异特征值)
			Args:
				sim_feature_matrix: 多模态融合特征相稀疏似度矩阵, 形状为(1024, 1024),对称,非负，稀疏化
			Return:
				最佳聚类数, Int类型 
		'''
		try:
			# 计算度矩阵D_matrix
			D_matrix = np.diag(np.sum(sim_feature_matrix, axis=1))
			D_matrix_inv_sqrt = np.linalg.inv(np.sqrt(D_matrix))
			# 构建归一化拉普拉斯矩阵Laplacian Matrix 
			#laplacian_matirx = D_matrix - sim_feature_matrix # 半正定、特征值[0, 无穷大]
			laplacian_matirx = np.eye(sim_feature_matrix.shape[0]) - D_matrix_inv_sqrt @ sim_feature_matrix @ D_matrix_inv_sqrt
			# 计算实对称矩阵的特征值
			features_values = np.linalg.eigvalsh(laplacian_matirx) # eigvalsh升序排列实对称矩阵的特征值：[1, 2, 3, 4,...]
			features_values = features_values[features_values > 1e-10] # 过滤0元素
			features_values_sort_topk = features_values[:self.sim_top_k] # 并取出前topk [1, 2, 3, 4,...]
			features_margin = np.diff(features_values_sort_topk) # 相邻特征值的间隔 [1, 1, ,1 ,1 ..., 1]
			max_margin_index = np.argmax(features_margin) # 找到最大特征值间隔所在的索引 
			best_cluster_num = max_margin_index + 1 # 最佳的聚类数 (因为此处的特征值之间有显著的差异)
			return best_cluster_num
		except np.linalg.LinAlgError:
			print("Matrix irreversibility")
			return None

class InterestDebiase(object):
	'''
		@Desc: 兴趣纠偏模块
	'''
	def __init__(self, origin_interaction_graph, generated_interaction_graph, interest_cluster_space_dict, image_modality, text_modality, audio_modality, sample_ratio):
		'''
			Args:
				origin_interaction_graph: 原始交互图，形状为(users, items)
				generated_interaction_graph: 生成交互图，形状同原始图
				interest_cluster_space_dict: 各模态下的物品聚类标签字典
				modality: 当前使用的模态
		'''
		self.origin_interaction_graph = origin_interaction_graph
		self.generated_interaction_graph = generated_interaction_graph
		self.interest_cluster_space_dict = interest_cluster_space_dict
		# self.modality = modality
		self.image_modality = image_modality
		self.text_modality = text_modality
		self.audio_modality = audio_modality
		self.sample_ratio = sample_ratio
		
		# 预处理用户兴趣数据（聚类集合、频率字典等）
		self.image_user_interest_map, self.text_user_interest_map, self.audio_user_interest_map = self.get_user_origin_interest_hash_map(
			origin_interaction_graph=origin_interaction_graph,
			interest_cluster_space_dict=interest_cluster_space_dict,
			image_modality=self.image_modality,
			text_modality=self.text_modality,
			audio_modality=self.audio_modality
		)
		#print("------------->InterestDebiase Start! ")
	def find_candidates(self):
		'''
			找出0->1和1->0变化的候选对
			Returns: (dislike_to_like, like_to_dislike) 各为(user_idx, item_idx)列表
			self.generated_interaction_graph: 
			tensor([[1., 1., 1.,  ..., 1., 0., 0.],
			[0., 1., 0.,  ..., 1., 1., 1.],
			[1., 0., 1.,  ..., 1., 1., 1.],
			...,
			[1., 1., 1.,  ..., 1., 1., 1.],
			[1., 1., 1.,  ..., 1., 1., 1.],
			[0., 0., 0.,  ..., 1., 0., 1.]], device='cuda:0')
			self.origin_interaction_graph: tensor([[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					...,
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0')
			flip_graph: tensor([[1., 1., 1.,  ..., 1., 0., 0.],
					[0., 1., 0.,  ..., 1., 1., 1.],
					[1., 0., 1.,  ..., 1., 1., 1.],
					...,
					[1., 1., 1.,  ..., 1., 1., 1.],
					[1., 1., 1.,  ..., 1., 1., 1.],
					[0., 0., 0.,  ..., 1., 0., 1.]], device='cuda:0')
		'''
		if self.generated_interaction_graph.shape != self.origin_interaction_graph.shape:
			raise ValueError("Shape mismatch between generated and origin graphs")
		
		# 计算差异矩阵 (非常的大,可以进行采样)
		# TODO 
		# print("self.generated_interaction_graph:", self.generated_interaction_graph)
		# print("self.origin_interaction_graph:", self.origin_interaction_graph)
		flip_graph = self.generated_interaction_graph - self.origin_interaction_graph
		# print("flip_graph:", flip_graph)
		# 获取变化索引（向量化操作）
		dislike_to_like = torch.where(flip_graph > 0)
		like_to_dislike = torch.where(flip_graph < 0)
		
		# 转换为Python列表
		dislike_pairs = list(zip(dislike_to_like[0].tolist(), dislike_to_like[1].tolist()))
		dislike_pairs = [(u, i) for u, i in dislike_pairs]  # 确保为int类型
		like_pairs = list(zip(like_to_dislike[0].tolist(), like_to_dislike[1].tolist()))
		like_pairs = [(u, i) for u, i in like_pairs]

		 # 随机采样逻辑
		import random
		# def sample_pairs(pairs, ratio):
		# 	if ratio >= 1.0:
		# 		return pairs
		# 	n = max(1, int(len(pairs) * ratio))
		# 	return random.sample(pairs, n)
		# print("interest sample rate:", self.sample_ratio)
		def safe_sample(pairs, ratio):
			# 输入保护
			if not isinstance(pairs, list) or ratio < 0:
				return []
			# 约束ratio范围
			safe_ratio = max(0.0, min(1.0, ratio))
			# 计算合法采样数量
			n = int(len(pairs) * safe_ratio)
			n = max(0, min(n, len(pairs)))
			if n == 0 or len(pairs) == 0:
				return []
			return random.sample(pairs, n)
		# return sample_pairs(dislike_pairs, self.sample_ratio), sample_pairs(like_pairs, self.sample_ratio)
		dislike_pairs_ , like_pairs_ = (safe_sample(dislike_pairs, self.sample_ratio),  safe_sample(like_pairs, self.sample_ratio))
		# print("dislike_to_like size:", len(dislike_pairs_ ), "like_to_dislike size:", len(like_pairs_))
		return dislike_pairs_ , like_pairs_ 
	


		
	def interest_query_debiase(self):
		'''
			执行兴趣纠偏，返回修正后的交互图
		'''
		# 初始化修正图为生成图的副本
		self.debiased_interaction_graph = self.generated_interaction_graph.clone()
		
		# 获取候选对
		dislike_pairs, like_pairs = self.find_candidates()
		
		# 处理0->1变化（潜在兴趣）
		for u, i in dislike_pairs:
			# 获取物品i的聚类标签
			item_cluster_image = self.interest_cluster_space_dict[self.image_modality][i]
			item_cluster_text = self.interest_cluster_space_dict[self.image_modality][i]
			if self.audio_modality is not None:
				item_cluster_audio = self.interest_cluster_space_dict[self.image_modality][i]
			# 判断用户u是否对该类感兴趣
			# user_data = self.fusion_space_origin_users_interest_hash_map.get(u, {'cluster_set': set()})
			user_data_image_hash = self.image_user_interest_map.get(u, {'cluster_set': set()})
			user_data_text_hash = self.image_user_interest_map.get(u, {'cluster_set': set()})
			if self.audio_modality is not None:
				user_data_audio_hash = self.image_user_interest_map.get(u, {'cluster_set': set()})
				if item_cluster_audio in user_data_audio_hash['cluster_set']:
					self.debiased_interaction_graph[u, i] = 1  # 保留
				else:
					self.debiased_interaction_graph[u, i] = 0  # 抑制

			if item_cluster_image in user_data_image_hash['cluster_set'] or item_cluster_text in user_data_text_hash['cluster_set']:
				self.debiased_interaction_graph[u, i] = 1  # 保留
			else:
				self.debiased_interaction_graph[u, i] = 0  # 抑制

		if len(like_pairs) > 0 :
			# 处理1->0变化（误点击）
			for u, i in like_pairs:
				# 获取当前物品的聚类及频率信息
				user_data_image_hash = self.image_user_interest_map.get(u, {'cluster_counts': set()})
				user_data_text_hash = self.image_user_interest_map.get(u, {'cluster_counts': set()})
				if self.audio_modality is not None:
					user_data_audio_hash = self.image_user_interest_map.get(u, {'cluster_counts': set()})
		
				item_cluster_image = self.interest_cluster_space_dict[self.image_modality][i]
				item_cluster_text = self.interest_cluster_space_dict[self.image_modality][i]
				if self.audio_modality is not None:
					item_cluster_audio = self.interest_cluster_space_dict[self.image_modality][i]

				# item_cluster_image = self.interest_cluster_space_dict[self.image_modality][i]
				# print("user_data_image_hash:", user_data_image_hash)
				cur_freq_image = user_data_image_hash['cluster_counts'].get(item_cluster_image, 0)
				cur_freq_text = user_data_text_hash['cluster_counts'].get(item_cluster_text, 0)
				if self.audio_modality is  not None:
					cur_freq_audio= user_data_audio_hash['cluster_counts'] .get(item_cluster_audio, 0)

				# 计算图像模态频率阈值
				counts = list(user_data_image_hash['cluster_counts'].values())
				if not counts:
					min_freq = mean_freq = 0
				else:
					min_freq = min(counts)
					mean_freq = sum(counts) / len(counts)
				# 决策逻辑
				if cur_freq_image <= (min_freq + 1):
					self.debiased_interaction_graph[u, i] = 0  # 保持抑制
				else:
					self.debiased_interaction_graph[u, i] = 1  # 恢复
		
				# 计算文本模态频率阈值
				counts = list(user_data_text_hash['cluster_counts'].values())
				if not counts:
					min_freq = mean_freq = 0
				else:
					min_freq = min(counts)
					mean_freq = sum(counts) / len(counts)
				# 决策逻辑
				if cur_freq_text <= (min_freq + 1):
					self.debiased_interaction_graph[u, i] = 0  # 保持抑制
				else:
					self.debiased_interaction_graph[u, i] = 1  # 恢复

				if self.audio_modality is not None:
					
					# 计算文本模态频率阈值
					counts = list(user_data_text_hash['cluster_counts'].values())
					if not counts:
						min_freq = mean_freq = 0
					else:
						min_freq = min(counts)
						mean_freq = sum(counts) / len(counts)
					# 决策逻辑
					if cur_freq_audio <= (min_freq + 1):
						self.debiased_interaction_graph[u, i] = 0  # 保持抑制
					else:
						self.debiased_interaction_graph[u, i] = 1  # 恢复


		return self.debiased_interaction_graph
	
	def get_user_origin_interest_hash_map(self, origin_interaction_graph, interest_cluster_space_dict, image_modality, text_modality, audio_modality=None):
		'''
			预处理用户兴趣数据（聚类集合、频率字典等）
			Returns: dict格式 {user_idx: {'cluster_set': set(), 'cluster_counts': dict()}}
		'''
		# 转换为numpy数组以加速处理
		if isinstance(origin_interaction_graph, torch.Tensor):
			origin_interaction_graph = origin_interaction_graph.cpu().numpy()
			
		image_modality_cluster_labels = np.array(interest_cluster_space_dict[image_modality])
		text_modality_cluster_labels = np.array(interest_cluster_space_dict[image_modality])
		if audio_modality is not None:
			audio_modality_cluster_labels = np.array(interest_cluster_space_dict[image_modality])	
		
		image_user_interest_map, text_user_interest_map, audio_user_interest_map = {}, {}, {}
		for u in range(origin_interaction_graph.shape[0]):
			# 找到用户u交互的物品索引
			interacted_items = np.where(origin_interaction_graph[u] > 0)[0]
			if len(interacted_items) == 0:
				image_user_interest_map[u] = {'cluster_set': set(), 'cluster_counts': {}}
				text_user_interest_map[u] = {'cluster_set': set(), 'cluster_counts': {}}
				audio_user_interest_map[u] = {'cluster_set': set(), 'cluster_counts': {}}
				continue

			# 向量化获取图像模态聚类标签并统计
			image_modality_clusters = image_modality_cluster_labels[interacted_items]
			unique_clusters, counts = np.unique(image_modality_clusters, return_counts=True)
			# 构建频率字典和集合
			cluster_counts = dict(zip(unique_clusters, counts))
			cluster_set = set(unique_clusters)
			image_user_interest_map[u] = {
				'cluster_set': cluster_set, # 当前用户喜欢的物品集合
				'cluster_counts': cluster_counts # 喜欢物品出现的频率
			}
		
			# 向量化获取图像模态聚类标签并统计
			text_modality_clusters = text_modality_cluster_labels[interacted_items]
			unique_clusters, counts = np.unique(text_modality_clusters, return_counts=True)
			# 构建频率字典和集合
			cluster_counts = dict(zip(unique_clusters, counts))
			cluster_set = set(unique_clusters)
			text_user_interest_map[u] = {
				'cluster_set': cluster_set, # 当前用户喜欢的物品集合
				'cluster_counts': cluster_counts # 喜欢物品出现的频率
			}
			if audio_modality is not None:
			# 向量化获取图像模态聚类标签并统计
				audio_modality_clusters = audio_modality_cluster_labels[interacted_items]
				unique_clusters, counts = np.unique(audio_modality_clusters, return_counts=True)
				# 构建频率字典和集合
				cluster_counts = dict(zip(unique_clusters, counts))
				cluster_set = set(unique_clusters)
				audio_user_interest_map[u] = {
					'cluster_set': cluster_set, # 当前用户喜欢的物品集合
					'cluster_counts': cluster_counts # 喜欢物品出现的频率
				}


		return image_user_interest_map, text_user_interest_map, audio_user_interest_map