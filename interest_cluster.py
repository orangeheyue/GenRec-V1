# Desc: Clustering for the multimodal features.
# Author: OrangeAI Research Team
# Time: 2025-03-24
# implement: GenRec-V1 | Flip is Better than Noise: Unbiased Interest Generation for Multimedia Recommendation


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
		Desc: Multimodal Feature Clustering
				Modal-Specific Clustering: Modal-Specific Clustering (clustering dedicated to individual modalities)
				Modal Fusion Clustering: Modal Fusion Clustering (clustering based on fused multimodal features)
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
			Desc:Cluster features of a specific modality using the K-means clustering method
			Args:
				specific_modal_features: Modal features. If there are 1024 items and the image feature dimension is 64, the shape is (1024, 64)
			Return:
				items_cluster_labels (cluster category labels assigned to each item ID), with a shape of (1024,). It stores an integer list ranging from 0 to (number of clusters - 1)
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
				Optimal number of clusters, Int type
			Attention:it only needs to be run once to obtain the optimal number of clusters in advance.
		'''
		print("--------------->auto get_kmeans_cluster_optimal_num:")
		# print("stand_feature_matrix:", stand_feature_matrix)
		distortions = [] # Store the within-cluster sum of squares corresponding to each k value.
		# k_range_max = stand_feature_matrix.shape[0] 
		# print("k_range_max:", k_range_max)
		for i in range(self.kmeans_cluster_num_min, self.kmeans_cluster_num_max, self.stride):
			kmeans_cluster = KMeans(n_clusters=i).fit(stand_feature_matrix)
			distortions.append(kmeans_cluster.inertia_) # 
		# Calculate the second-order difference (to find the mutation point of the Inertia descending speed, i.e., the point with the maximum curvature)
		print("distortions:", distortions)
		diff2 = np.diff(np.diff(distortions))
		# print("np.argmin(diff2):", np.argmin(diff2))
		best_cluster_nums = np.argmin(diff2) + self.kmeans_cluster_num_min + 1 
		return best_cluster_nums

	def multimodal_fusion_cluster(self, fusion_modal_feature):
		'''
			Desc: Cluster multimodal fused features using the spectral clustering method
			Args:
				fusion_modal_feature: Multimodal fused features, i.e., features processed by the multimodal Fusion module, with the shape remaining (1024, 64)
			Return:
				items_cluster_labels (cluster category labels assigned to each item ID), with a shape of (1024,).
				It stores an integer list ranging from 0 to (number of clusters - 1) (the item categories preferred by each user)
		'''
		sim_feature_matrix = cosine_similarity(fusion_modal_feature)
		# Sparsification of the similarity matrix, aiming to retain only the top-k most similar ones
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
			(Obtain the optimal number of clusters for spectral clustering using the Spectral Gap method, which essentially involves identifying eigenvalues with significant differences)
			Args:
				sim_feature_matrix: Sparse similarity matrix of multimodal fused features, with a shape of (1024, 1024). It is symmetric, non-negative, and sparsified.
			Return:
				Optimal number of clusters, Int type
		'''
		try:
			# Calculate the Degree Matrix D_matrix
			D_matrix = np.diag(np.sum(sim_feature_matrix, axis=1))
			D_matrix_inv_sqrt = np.linalg.inv(np.sqrt(D_matrix))
			# Build Laplacian Matrix 
			#laplacian_matirx = D_matrix - sim_feature_matrix 
			laplacian_matirx = np.eye(sim_feature_matrix.shape[0]) - D_matrix_inv_sqrt @ sim_feature_matrix @ D_matrix_inv_sqrt
			# Calculate the eigenvalues of a real symmetric matrix
			features_values = np.linalg.eigvalsh(laplacian_matirx) 
			features_values = features_values[features_values > 1e-10] 
			features_values_sort_topk = features_values[:self.sim_top_k] 
			features_margin = np.diff(features_values_sort_topk) 
			max_margin_index = np.argmax(features_margin) 
			best_cluster_num = max_margin_index + 1 # best cluster num
			return best_cluster_num
		except np.linalg.LinAlgError:
			print("Matrix irreversibility")
			return None


class InterestDebiase(object):
	'''
		@Desc: InterestDebiase
	'''
	def __init__(self, origin_interaction_graph, generated_interaction_graph, interest_cluster_space_dict, image_modality, text_modality, audio_modality, sample_ratio):
		'''
			Args:
				origin_interaction_graph: (users, items)
				generated_interaction_graph: 
				interest_cluster_space_dict: 
				modality: 
		'''
		self.origin_interaction_graph = origin_interaction_graph
		self.generated_interaction_graph = generated_interaction_graph
		self.interest_cluster_space_dict = interest_cluster_space_dict
		# self.modality = modality
		self.image_modality = image_modality
		self.text_modality = text_modality
		self.audio_modality = audio_modality
		self.sample_ratio = sample_ratio
		
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
			0->1, 1->0
			Returns: (dislike_to_like, like_to_dislike) (user_idx, item_idx) lists
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
		# print("self.generated_interaction_graph:", self.generated_interaction_graph)
		# print("self.origin_interaction_graph:", self.origin_interaction_graph)
		flip_graph = self.generated_interaction_graph - self.origin_interaction_graph
		# print("flip_graph:", flip_graph)
		dislike_to_like = torch.where(flip_graph > 0)
		like_to_dislike = torch.where(flip_graph < 0)

		dislike_pairs = list(zip(dislike_to_like[0].tolist(), dislike_to_like[1].tolist()))
		dislike_pairs = [(u, i) for u, i in dislike_pairs]  
		like_pairs = list(zip(like_to_dislike[0].tolist(), like_to_dislike[1].tolist()))
		like_pairs = [(u, i) for u, i in like_pairs]

		import random
		# def sample_pairs(pairs, ratio):
		# 	if ratio >= 1.0:
		# 		return pairs
		# 	n = max(1, int(len(pairs) * ratio))
		# 	return random.sample(pairs, n)
		# print("interest sample rate:", self.sample_ratio)
		def safe_sample(pairs, ratio):
			if not isinstance(pairs, list) or ratio < 0:
				return []
			safe_ratio = max(0.0, min(1.0, ratio))
			n = int(len(pairs) * safe_ratio)
			n = max(0, min(n, len(pairs)))
			if n == 0 or len(pairs) == 0:
				return []
			return random.sample(pairs, n)
		dislike_pairs_ , like_pairs_ = (safe_sample(dislike_pairs, self.sample_ratio),  safe_sample(like_pairs, self.sample_ratio))
		# print("dislike_to_like size:", len(dislike_pairs_ ), "like_to_dislike size:", len(like_pairs_))
		return dislike_pairs_ , like_pairs_ 
	
	
	def interest_query_debiase(self):
		'''
			Perform interest rectification and return the corrected interaction graph
		'''
		self.debiased_interaction_graph = self.generated_interaction_graph.clone()
		dislike_pairs, like_pairs = self.find_candidates()
		
		# Handle the 0->1 change (latent interest)
		for u, i in dislike_pairs:
			# Get the cluster label of item i
			item_cluster_image = self.interest_cluster_space_dict[self.image_modality][i]
			item_cluster_text = self.interest_cluster_space_dict[self.image_modality][i]
			if self.audio_modality is not None:
				item_cluster_audio = self.interest_cluster_space_dict[self.image_modality][i]
			# Determine whether user u is interested in this category
			# user_data = self.fusion_space_origin_users_interest_hash_map.get(u, {'cluster_set': set()})
			user_data_image_hash = self.image_user_interest_map.get(u, {'cluster_set': set()})
			user_data_text_hash = self.image_user_interest_map.get(u, {'cluster_set': set()})
			if self.audio_modality is not None:
				user_data_audio_hash = self.image_user_interest_map.get(u, {'cluster_set': set()})
				if item_cluster_audio in user_data_audio_hash['cluster_set']:
					self.debiased_interaction_graph[u, i] = 1  # keep
				else:
					self.debiased_interaction_graph[u, i] = 0  # suppress interference signals

			if item_cluster_image in user_data_image_hash['cluster_set'] or item_cluster_text in user_data_text_hash['cluster_set']:
				self.debiased_interaction_graph[u, i] = 1  
			else:
				self.debiased_interaction_graph[u, i] = 0 
		if len(like_pairs) > 0 :
			# Handle the 1->0 change (accidental click)
			for u, i in like_pairs:
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
				counts = list(user_data_image_hash['cluster_counts'].values())
				if not counts:
					min_freq = mean_freq = 0
				else:
					min_freq = min(counts)
					mean_freq = sum(counts) / len(counts)
				# dicision logic
				if cur_freq_image <= (min_freq + 1):
					self.debiased_interaction_graph[u, i] = 0  # keep
				else:
					self.debiased_interaction_graph[u, i] = 1  # suppress

				counts = list(user_data_text_hash['cluster_counts'].values())
				if not counts:
					min_freq = mean_freq = 0
				else:
					min_freq = min(counts)
					mean_freq = sum(counts) / len(counts)
				# dicision logic
				if cur_freq_text <= (min_freq + 1):
					self.debiased_interaction_graph[u, i] = 0  # keep
				else:
					self.debiased_interaction_graph[u, i] = 1  # recover

				if self.audio_modality is not None:
					counts = list(user_data_text_hash['cluster_counts'].values())
					if not counts:
						min_freq = mean_freq = 0
					else:
						min_freq = min(counts)
						mean_freq = sum(counts) / len(counts)
					if cur_freq_audio <= (min_freq + 1):
						self.debiased_interaction_graph[u, i] = 0  # keep
					else:
						self.debiased_interaction_graph[u, i] = 1  # recover

		return self.debiased_interaction_graph
	
	def get_user_origin_interest_hash_map(self, origin_interaction_graph, interest_cluster_space_dict, image_modality, text_modality, audio_modality=None):
		'''
			Preprocess user interest data (including cluster sets, frequency dictionaries, etc.)
			Returns: dict格式 {user_idx: {'cluster_set': set(), 'cluster_counts': dict()}}
		'''
		if isinstance(origin_interaction_graph, torch.Tensor):
			origin_interaction_graph = origin_interaction_graph.cpu().numpy()
			
		image_modality_cluster_labels = np.array(interest_cluster_space_dict[image_modality])
		text_modality_cluster_labels = np.array(interest_cluster_space_dict[image_modality])
		if audio_modality is not None:
			audio_modality_cluster_labels = np.array(interest_cluster_space_dict[image_modality])	
		
		image_user_interest_map, text_user_interest_map, audio_user_interest_map = {}, {}, {}
		for u in range(origin_interaction_graph.shape[0]):
			interacted_items = np.where(origin_interaction_graph[u] > 0)[0]
			if len(interacted_items) == 0:
				image_user_interest_map[u] = {'cluster_set': set(), 'cluster_counts': {}}
				text_user_interest_map[u] = {'cluster_set': set(), 'cluster_counts': {}}
				audio_user_interest_map[u] = {'cluster_set': set(), 'cluster_counts': {}}
				continue

			image_modality_clusters = image_modality_cluster_labels[interacted_items]
			unique_clusters, counts = np.unique(image_modality_clusters, return_counts=True)
			cluster_counts = dict(zip(unique_clusters, counts))
			cluster_set = set(unique_clusters)
			image_user_interest_map[u] = {
				'cluster_set': cluster_set, # The set of items preferred by the current user
				'cluster_counts': cluster_counts # The occurrence frequency of preferred items
			}
		
			text_modality_clusters = text_modality_cluster_labels[interacted_items]
			unique_clusters, counts = np.unique(text_modality_clusters, return_counts=True)
			cluster_counts = dict(zip(unique_clusters, counts))
			cluster_set = set(unique_clusters)
			text_user_interest_map[u] = {
				'cluster_set': cluster_set, 
				'cluster_counts': cluster_counts 
			}
			if audio_modality is not None:
				audio_modality_clusters = audio_modality_cluster_labels[interacted_items]
				unique_clusters, counts = np.unique(audio_modality_clusters, return_counts=True)
				cluster_counts = dict(zip(unique_clusters, counts))
				cluster_set = set(unique_clusters)
				audio_user_interest_map[u] = {
					'cluster_set': cluster_set,
					'cluster_counts': cluster_counts
				}

		return image_user_interest_map, text_user_interest_map, audio_user_interest_map