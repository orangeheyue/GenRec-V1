import torch
from torch.utils.tensorboard import SummaryWriter
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import  GCNModel, ModalDenoiseTransformer, FlipInterestDiffusion
from interest_cluster import MultimodalCluster, InterestDebiase

from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp
import random
import math
import setproctitle
from scipy.sparse import coo_matrix
import json
from debiased_rate import DebiasingMetrics

class Coach:
	def __init__(self, handler):
		self.writer = SummaryWriter('runs/experiment')
		self.handler = handler

		self.modal_fusion = False

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')

		recallMax = 0
		ndcgMax = 0
		precisionMax = 0
		bestEpoch = 0

		log('Model Initialized')

		for ep in range(0, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch(ep)
			# self.scheduler.step()
			log(self.makePrint('Train', ep, reses, tstFlag))
			self.writer.add_scalar('Loss/Train', reses['Loss'], ep)
			self.writer.add_scalar('Loss/BPR', reses['BPR Loss'], ep)
			self.writer.add_scalar('Loss/CL', reses['CL loss'], ep)
			
			if tstFlag:
				reses = self.testEpoch()
				if (reses['Recall'] > recallMax):
					recallMax = reses['Recall']
					ndcgMax = reses['NDCG']
					precisionMax = reses['Precision']
					bestEpoch = ep
				log(self.makePrint('Test', ep, reses, tstFlag))
				self.writer.add_scalar('Metric/Recall', reses['Recall'], ep)
				self.writer.add_scalar('Metric/NDCG', reses['NDCG'], ep)
				self.writer.add_scalar('Metric/Precision', reses['Precision'], ep)
		print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax, ' , Precision', precisionMax)
		self.writer.close()

	def prepareModel(self):
		if args.data == 'tiktok':
			self.image_embedding, self.text_embedding, self.audio_embedding = self.handler.image_feats.detach(), self.handler.text_feats.detach(), self.handler.audio_feats.detach()
			self.model = GCNModel(self.handler.image_feats.detach(), self.handler.text_feats.detach(), self.handler.audio_feats.detach(), modal_fusion=self.modal_fusion).cuda()
		else:
			self.image_embedding, self.text_embedding = self.handler.image_feats.detach(), self.handler.text_feats.detach()
			self.model = GCNModel(self.handler.image_feats.detach(), self.handler.text_feats.detach(), modal_fusion=self.modal_fusion).cuda()
		self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
		
		# get user`s interest cluster space
		debiased_denoised_batch = None 
		image_modal_cluster = MultimodalCluster(
			num_cluster_visual_modal=20,
			num_cluster_text_modal=20,
			num_cluster_audio_modal=20,
			num_cluster_fusion_modal=20,
			kmeans_cluster_num=args.kmeans_cluster_num,
			spectral_cluster_num=20,
			sim_top_k=20,
			use_auto_optimal_k=args.use_auto_optimal_k,
			kmeans_cluster_num_min=3,   #TikTok: interest_min: 0 interest_mean: 7 interest_max: 603
			kmeans_cluster_num_mean=7,  # Baby: interest_min: 3 interest_mean: 7 interest_max: 100
			kmeans_cluster_num_max=237, # Sport: interest_min: 3 interest_mean: 7 interest_max: 237
			kmeans_stride=10
		)

		image_modal_optimial_cluster_num, text_optimial_cluster_num, audio_optimial_cluster_num = 18, 59, 46 
		if args.use_auto_optimal_k == False and args.data == 'tiktok':
			image_modal_optimial_cluster_num = 18# 18
			text_optimial_cluster_num = 59 #59
			audio_optimial_cluster_num = 46 #46

		if args.use_auto_optimal_k == False and args.data == 'baby':
			image_modal_optimial_cluster_num = 6 #6
			text_optimial_cluster_num = 11 # 11

		if args.use_auto_optimal_k == False and args.data == 'sports':
			image_modal_optimial_cluster_num = 9
			text_optimial_cluster_num = 12

		image_modal_items_cluster_labels = image_modal_cluster.multimodal_specific_cluster(specific_modal_features=self.image_embedding, modality='image_modal', optimial_cluster_num=image_modal_optimial_cluster_num)
		print("image_modal_items_cluster_labels:", image_modal_items_cluster_labels, "image_modal_items_cluster_labels:", len(image_modal_items_cluster_labels))
		text_modal_items_cluster_labels = image_modal_cluster.multimodal_specific_cluster(specific_modal_features=self.text_embedding, modality='text_modal', optimial_cluster_num=text_optimial_cluster_num)
		print("text_modal_items_cluster_labels:", text_modal_items_cluster_labels, "image_modal_items_cluster_labels:", len(text_modal_items_cluster_labels))
		if args.data == 'tiktok':
			audio_modal_items_cluster_labels = image_modal_cluster.multimodal_specific_cluster(specific_modal_features=self.audio_embedding, modality='audio_modal', optimial_cluster_num=audio_optimial_cluster_num)
			print("audio_modal_items_cluster_labels:", audio_modal_items_cluster_labels, "image_modal_items_cluster_labels:", len(audio_modal_items_cluster_labels))
			self.multimodal_interest_space = {
				'image_modal' : image_modal_items_cluster_labels,
				'text_modal' : text_modal_items_cluster_labels,
				'audio_modal' : audio_modal_items_cluster_labels
			}
		else:
			self.multimodal_interest_space = {
				'image_modal' : image_modal_items_cluster_labels,
				'text_modal' : text_modal_items_cluster_labels
			}
		INTEREST_SPACE_SAVE_PATH = './multimodal_interest_space.json'

		# Item-Item matrix 
		'''
			self.image_II_matrix.shape: torch.Size([6710, 6710])
			self.text_II_matrix.shape: torch.Size([6710, 6710])
			self.audio_II_matrix.shape: torch.Size([6710, 6710])
		'''
		self.image_II_origin_matrix_dense, self.image_II_matrix= self.buildItem2ItemMatrix(self.image_embedding) 
		self.text_II_origin_matrix_dense, self.text_II_matrix = self.buildItem2ItemMatrix(self.text_embedding)	
		if args.data == 'tiktok':
			self.audio_II_origin_matrix_dense, self.audio_II_matrix = self.buildItem2ItemMatrix(self.audio_embedding)	

		self.diffusion_model = FlipInterestDiffusion(
			steps=args.steps,
			base_temp=args.flip_temp
		)

		out_dims = self.image_embedding.shape[0]
		in_dims = self.image_embedding.shape[0]
		#self.denoise_model_image = ModalDenoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()

		self.denoise_model_image = ModalDenoiseTransformer(in_dims=in_dims
													 ,out_dims=out_dims
													 ,emb_size=args.d_emb_size
													 ,nhead=args.nhead 
													 ,num_layers=args.num_layers
													 ).cuda()
		self.denoise_opt_image = torch.optim.Adam(self.denoise_model_image.parameters(), lr=args.lr, weight_decay=0)

	def normalizeAdj(self, mat): 
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def buildUIMatrix(self, u_list, i_list, edge_list):
		mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32)

		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)

		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()
	
	def buildItem2ItemMatrix(self, feature):
		'''
			modality guided item-item similarity matrix
		'''
		feature_embedding = torch.nn.Embedding.from_pretrained(feature, freeze=False)
		feature_embedding = feature_embedding.weight.detach()
		feature_norm = feature.div(torch.norm(feature_embedding, p=2, dim=-1, keepdim=True))
		sim_adj = torch.mm(feature_norm, feature_norm.transpose(1, 0))
		sim_adj_sparse = build_knn_normalized_graph(sim_adj, topk=args.knn_k, is_sparse=args.sparse, norm_type='sym')
		
		return sim_adj, sim_adj_sparse
		
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

	def trainEpoch(self, ep):
		'''
			Generative training for User-Item Graph
		'''
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		epLoss, epRecLoss, epClLoss = 0, 0, 0
		# add Bias_Rate, KL_Rate
		bias_rate_without_mid, kl_rate_without_mid, bias_rate_with_mid, kl_rate_with_mid = 0, 0, 0, 0
		
		epDiLoss = 0
		epDiLoss_image, epDiLoss_text = 0, 0
		if args.data == 'tiktok':
			epDiLoss_audio = 0

		steps = trnLoader.dataset.__len__() // args.batch

		diffusionLoader = self.handler.diffusionLoader

		# FIND the Cluster Range 
		train_epoch_num = 0 
		train_batch_num = 0
		user_interest_num_min_list, user_interest_num_mid_list, user_interest_num_max_list = [], [], []

		for i, batch in enumerate(diffusionLoader):
			'''
			batch: [tensor([[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					...,
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.]]), tensor([5262, 5684, 5341,  ..., 2287, 5615, 1071])]
				batch_item.shape: torch.Size([1024, 6710])
				batch_index.shape: torch.Size([1024])
			'''
			train_batch_num += 1
			# load diffusion dataset like user-item interaction graph.
			batch_item, batch_index = batch
			# print("batch_item.shape:", batch_item.shape)
			# print("batch_index.shape:", batch_index.shape)
			batch_item, batch_index = batch_item.cuda(), batch_index.cuda()
			
			###################################FIND The Cluster Range###################################
			# Baby: interest_min: 3 interest_mean: 7 interest_max: 100
			# TikTok: interest_min: 0 interest_mean: 7 interest_max: 603
			# Sport: interest_min: 3 interest_mean: 7 interest_max: 237
			###################################FIND The Cluster Range###################################

			row_sum = batch_item.sum(dim=1)
			user_interest_num_min, user_interest_num_mid, user_interest_num_max = row_sum.min().item() , row_sum.mean().item() , row_sum.max().item()
			# print("user_interest_num_min:", user_interest_num_min, "user_interest_num_mid:", user_interest_num_mid, "user_interest_num_max:", user_interest_num_max)
			user_interest_num_min_list.append(user_interest_num_min)
			user_interest_num_mid_list.append(user_interest_num_mid)
			user_interest_num_max_list.append(user_interest_num_max)

			iEmbeds = self.model.getItemEmbeds().detach()
			uEmbeds = self.model.getUserEmbeds().detach()

			image_feats = self.model.getImageFeats().detach()
			text_feats = self.model.getTextFeats().detach()
			audio_feats = None
			if args.data == 'tiktok':
				audio_feats = self.model.getAudioFeats().detach()
			#print("image_feats:", image_feats, "image_feats.shape:", image_feats.shape, "text_feats.shape:", text_feats.shape, "audio_feats.shape:", audio_feats.shape)
			self.denoise_opt_image.zero_grad()
			# self.denoise_opt_text.zero_grad()
			# if args.data == 'tiktok':
			# 	self.denoise_opt_audio.zero_grad()
			'''
				image_feats.shape: torch.Size([6710, 64]), text_feats.shape: torch.Size([6710, 64]), audio_feats.shape: torch.Size([6710, 64])
			'''
			# print("image_feats:", image_feats)
			# print("image_feats.shape:", image_feats.shape)
			# print("text_feats:", image_feats)
			# print("text_feats.shape:", image_feats.shape)
			# print("audio_feats:", image_feats)
			# print("audio_feats.shape:", image_feats.shape)
			loss_image = self.diffusion_model.training_losses(self.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats, text_feats, audio_feats)
			epDiLoss_image += loss_image.item()
			loss = loss_image
			loss.backward()
			self.denoise_opt_image.step()
			log('Diffusion Step %d/%d' % (i, diffusionLoader.dataset.__len__() // args.batch), save=False, oneline=True)
		log('')
		log('Start to re-build UI matrix')

		
		print("interest_min:",math.ceil(min(user_interest_num_min_list)), "interest_mean:", math.ceil(sum(user_interest_num_mid_list) / train_batch_num), "interest_max:", math.ceil(max(user_interest_num_max_list)))

		with torch.no_grad():
			'''
				Generative Infer to build an interest enhanced graph.
			'''
			u_list_image = []
			i_list_image = []
			edge_list_image = []

			u_list_text = []
			i_list_text = []
			edge_list_text = []

			if args.data == 'tiktok':
				u_list_audio = []
				i_list_audio = []
				edge_list_audio = []

			for batch_id, batch in enumerate(diffusionLoader):
				batch_item, batch_index = batch
				batch_item, batch_index = batch_item.cuda(), batch_index.cuda()
				#print("origin_UI_Interaction:", batch_item)

				'''
					origin_UI_Interaction: tensor([[0., 0., 0.,  ..., 0., 0., 0.],
										[0., 0., 0.,  ..., 0., 0., 0.],
										[0., 0., 0.,  ..., 0., 0., 0.],
										...,
										[0., 0., 0.,  ..., 0., 0., 0.],
										[0., 0., 0.,  ..., 0., 0., 0.],
										[0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:1')
					denoised_UI_Interaction: tensor([[ 0.1082,  0.1071,  0.0995,  ..., -0.0013,  0.1061,  0.1128],
											[ 0.0972,  0.0975,  0.0898,  ..., -0.0013,  0.0932,  0.1052],
											[ 0.1034,  0.1058,  0.0997,  ..., -0.0012,  0.1077,  0.1114],
											...,
											[ 0.1043,  0.0984,  0.1045,  ..., -0.0011,  0.0993,  0.1005],
											[ 0.1005,  0.1031,  0.0947,  ..., -0.0013,  0.1005,  0.0974],
											[ 0.0859,  0.0922,  0.1015,  ..., -0.0013,  0.0932,  0.1038]],
										device='cuda:1')
				'''
				# image
				#print("diffusion:", denoised_batch, "denoised_batch.shape:", denoised_batch.shape)
				'''
					denoised_batch tensor([[-0.0065,  0.0102,  0.0007,  ..., -0.0208,  0.0343, -0.0176],
						[-0.0034,  0.0129, -0.0017,  ..., -0.0308,  0.0387, -0.0202],
						[-0.0034,  0.0091,  0.0051,  ..., -0.0229,  0.0348, -0.0109],
						...,
						[ 0.0031,  0.0098,  0.0059,  ..., -0.0164,  0.0391, -0.0168],
						[-0.0032,  0.0146, -0.0023,  ..., -0.0251,  0.0329, -0.0150],
						[ 0.0014,  0.0059, -0.0020,  ..., -0.0235,  0.0317, -0.0109]],
					device='cuda:0')

					top_item: 
					tensor([[0.0619, 0.0584, 0.0567,  ..., 0.0518, 0.0499, 0.0493],
					[0.0667, 0.0618, 0.0604,  ..., 0.0505, 0.0503, 0.0493],
					[0.0668, 0.0637, 0.0608,  ..., 0.0502, 0.0502, 0.0500],
					...,
					[0.0653, 0.0624, 0.0621,  ..., 0.0499, 0.0498, 0.0493],
					[0.0666, 0.0651, 0.0591,  ..., 0.0543, 0.0512, 0.0511],
					[0.0644, 0.0603, 0.0576,  ..., 0.0527, 0.0518, 0.0517]],
					device='cuda:0')
				 
				   indices_: 
				   tensor([[4679, 5724,  964,  ...,  535, 2544, 1084],
					[5724, 4679, 2106,  ...,  964, 5668,  535],
					[5724, 4679, 6520,  ..., 4812, 2995,  964],
					...,
					[4679, 6520, 5724,  ..., 3543, 2564,  964],
					[5724, 6520, 4679,  ..., 2995, 2544, 5340],
					[5724, 6520, 4679,  ..., 1899, 4812, 2106]], device='cuda:0')


					self.image_II_matrix.shape: torch.Size([6710, 6710])
					self.text_II_matrix.shape: torch.Size([6710, 6710])
					self.audio_II_matrix.shape: torch.Size([6710, 6710])

				'''
				denoised_batch, denoised_prob = self.diffusion_model.p_sample(self.denoise_model_image, batch_item, args.sampling_steps, args.bayesian_samplinge_schedule)
				# denoised_batch += batch_item
				# print("batch_item:", batch_item)
				# print("denoised_batch:", denoised_batch) 
				# print("denoised_prob:", denoised_prob)
				_, indices = torch.topk(denoised_prob, k=args.gen_topk, dim=1) 
				mask = torch.zeros_like(denoised_prob, dtype=torch.bool).scatter_(1, indices, True)
				denoised_batch = torch.where(mask, denoised_batch, batch_item)
				# print("denoised_batch_top-k:", denoised_batch) 
				#print("denoised_batch:", denoised_batch, "batch_item:", batch_item, "diff:", (torch.sqrt(torch.pow(denoised_batch - batch_item, 2))).sum())
				if args.OpenInterestDebiase == True:
					# log("-------------------------->Open InterestDebiase Module------------------")
					if  args.data == 'tiktok':
						
						image_interest_judge = InterestDebiase(
						origin_interaction_graph = batch_item,
						generated_interaction_graph = denoised_batch,
						interest_cluster_space_dict = self.multimodal_interest_space,
						image_modality='image_modal',
						text_modality='text_modal',
						audio_modality='audio_modal',
						sample_ratio = args.sample_ratio 
						)

						results_without_mid = DebiasingMetrics().evaluate_debiasing_effect(denoised_batch.detach().cpu().numpy(), batch_item.detach().cpu().numpy())
						bias_rate_without_mid += results_without_mid['bias_rate']
						kl_rate_without_mid += results_without_mid['kl_divergence'] 
						denoised_batch = image_interest_judge.interest_query_debiase()
						results_with_mid = DebiasingMetrics().evaluate_debiasing_effect(denoised_batch.detach().cpu().numpy(), batch_item.detach().cpu().numpy())
						bias_rate_with_mid  += results_with_mid['bias_rate']
						kl_rate_with_mid  += results_with_mid['kl_divergence']
					else: 
						image_interest_judge = InterestDebiase(
						origin_interaction_graph = batch_item,
						generated_interaction_graph = denoised_batch,
						interest_cluster_space_dict = self.multimodal_interest_space,
						image_modality='image_modal',
						text_modality='text_modal',
						audio_modality=None,
						sample_ratio = args.sample_ratio 
						)
						
						results_without_mid = DebiasingMetrics().evaluate_debiasing_effect(denoised_batch.detach().cpu().numpy(), batch_item.detach().cpu().numpy())
						bias_rate_without_mid += results_without_mid['bias_rate']
						kl_rate_without_mid += results_without_mid['kl_divergence'] 
						denoised_batch = image_interest_judge.interest_query_debiase()
						results_with_mid = DebiasingMetrics().evaluate_debiasing_effect(denoised_batch.detach().cpu().numpy(), batch_item.detach().cpu().numpy())
						bias_rate_with_mid += results_with_mid['bias_rate']
						kl_rate_with_mid  += results_with_mid['kl_divergence']
				# else:
				# 	log("-------------------------->Attention!! You Don`t Open The InterestDebiase Module------------------")

				# print("batch_item:", batch_item)
				# print("denoised_batch:", denoised_batch) 
				# print("denoised_prob:", denoised_prob)
				# print("denoised_batch * denoised_prob:", denoised_batch * denoised_prob)
				top_item, indices_ = torch.topk(denoised_batch * denoised_prob, k=args.rebuild_k)
				# top_item, indices_ = torch.topk(denoised_batch*denoised_prob, k=args.rebuild_k)
				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]): 
						u_list_image.append(int(batch_index[i].cpu().numpy()))
						i_list_image.append(int(indices_[i][j].cpu().numpy()))
						edge_list_image.append(1.0)

	
			# image
			u_list_image = np.array(u_list_image)
			i_list_image = np.array(i_list_image)
			edge_list_image = np.array(edge_list_image)
			self.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
			self.image_UI_matrix = self.model.edgeDropper(self.image_UI_matrix)

			# # # text
			# u_list_text = np.array(u_list_text)
			# i_list_text = np.array(i_list_text)
			# edge_list_text = np.array(edge_list_text)
			# self.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
			# self.text_UI_matrix = self.model.edgeDropper(self.text_UI_matrix)

			# if args.data == 'tiktok':
			# 	# audio
			# 	u_list_audio = np.array(u_list_audio)
			# 	i_list_audio = np.array(i_list_audio)
			# 	edge_list_audio = np.array(edge_list_audio)
			# 	self.audio_UI_matrix = self.buildUIMatrix(u_list_audio, i_list_audio, edge_list_audio)
			# 	self.audio_UI_matrix = self.model.edgeDropper(self.audio_UI_matrix)

		log('UI matrix built!')

		# GCN 和对比学习
		for i, tem in enumerate(trnLoader):

			'''
				ancs: tensor([  10, 4153,  334,  ...,  117,  125,   24], device='cuda:0')    user id
				poss: tensor([3550,  521,   96,  ..., 2457, 2226, 4632], device='cuda:0')    item id
				negs: tensor([4984,  666, 2158,  ..., 6698, 5829, 4554], device='cuda:0')    neg item id
				ancs.shape: torch.Size([1024]) poss.shape: torch.Size([1024]) negs.shape: torch.Size([1024])

				self.handler.torchBiAdj:  (user+item, user+item) sparse adj matrix torchBiAdj.shape: torch.Size([16018, 16018])
					tensor(indices=tensor([[    0, 10193, 10695,  ..., 16015, 16016, 16017],
										[    0,     0,     0,  ..., 16015, 16016, 16017]]),
						values=tensor([0.2500, 0.1443, 0.0606,  ..., 1.0000, 1.0000, 1.0000]),
						device='cuda:0', size=(16018, 16018), nnz=135100, layout=torch.sparse_coo)
						self.handler.torchBiAdj.shape: torch.Size([16018, 16018])
			
				self.image_UI_matrix: tensor(indices=tensor([[14014, 14666, 10909,  ..., 16015, 16016, 16017],
									[    0,     1,     2,  ..., 16015, 16016, 16017]]),
					values=tensor([0.0311, 0.0766, 0.7071,  ..., 2.0000, 2.0000, 2.0000]),
					device='cuda:0', size=(16018, 16018), nnz=17221, layout=torch.sparse_coo)
				self.image_UI_matrix.shape: torch.Size([16018, 16018])
				self.text_UI_matrix: tensor(indices=tensor([[    0, 10024,     1,  ..., 16006, 16010, 16012],
									[    0,     0,     1,  ..., 16006, 16010, 16012]]),
					values=tensor([1.0000, 0.0504, 1.0000,  ..., 2.0000, 2.0000, 2.0000]),
					device='cuda:0', size=(16018, 16018), nnz=17359, layout=torch.sparse_coo)
				self.text_UI_matrix.shape: torch.Size([16018, 16018])
				self.audio_UI_matrix: tensor(indices=tensor([[15021,     1,  9777,  ..., 16014, 16015, 16017],
									[    0,     1,     1,  ..., 16014, 16015, 16017]]),
					values=tensor([0.6325, 1.0000, 0.0327,  ..., 2.0000, 2.0000, 2.0000]),
					device='cuda:0', size=(16018, 16018), nnz=17177, layout=torch.sparse_coo)
				self.audio_UI_matrix.shape: torch.Size([16018, 16018])
			'''
			ancs, poss, negs = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			negs = negs.long().cuda()


			self.opt.zero_grad()

			if args.data == 'tiktok':
				usrEmbeds, itmEmbeds, side_Embeds, content_Emebeds = self.model.forward(self.handler.R, self.handler.torchBiAdj, self.image_UI_matrix, self.image_II_matrix, self.text_II_matrix, self.audio_II_matrix, None) 
			else:
				self.audio_UI_matrix, self.audio_II_matrix = None, None
				usrEmbeds, itmEmbeds, side_Embeds, content_Emebeds = self.model.forward(self.handler.R, self.handler.torchBiAdj, self.image_UI_matrix, self.image_II_matrix, self.text_II_matrix, self.audio_II_matrix, None) 

			# Caculate Loss
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			negEmbeds = itmEmbeds[negs]

			bprLoss, _,  regLoss = self.model.bpr_loss(ancEmbeds, posEmbeds, negEmbeds)
			loss = bprLoss + regLoss
			epRecLoss += bprLoss.item()
			epLoss += loss.item()

			# caculate contrastive los
			side_embeds_users, side_embeds_items = torch.split(side_Embeds, [args.user, args.item], dim=0)
			content_embeds_user, content_embeds_items = torch.split(content_Emebeds, [args.user, args.item], dim=0)

			# item-item contrastive loss
			clLoss1 = self.model.infoNCE_loss(side_embeds_items[poss], content_embeds_items[poss], args.temp) +  self.model.infoNCE_loss(side_embeds_users[ancs], content_embeds_user[ancs], args.temp) 
			# user-item contrastive loss
			clLoss2 = self.model.infoNCE_loss(usrEmbeds[ancs], content_embeds_items[poss], args.temp) +  self.model.infoNCE_loss(usrEmbeds[ancs], side_embeds_items[poss], args.temp) 

			clLoss = clLoss1 * args.ssl_reg1  + clLoss2 * args.ssl_reg2

			loss += clLoss
			epClLoss += clLoss.item()

			loss.backward()
			self.opt.step()

			log('Step %d/%d: bpr : %.3f ; reg : %.3f ; cl : %.3f ' % (
				i, 
				steps,
				bprLoss.item(),
		regLoss.item(),
				clLoss.item()
				), save=False, oneline=True)

		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['BPR Loss'] = epRecLoss / steps
		ret['CL loss'] = epClLoss / steps
		ret['Di image loss'] = epDiLoss_image / (diffusionLoader.dataset.__len__() // args.batch)
		ret['Di text loss'] = epDiLoss_text / (diffusionLoader.dataset.__len__() // args.batch)
		if args.data == 'tiktok':
			ret['Di audio loss'] = epDiLoss_audio / (diffusionLoader.dataset.__len__() // args.batch)
		ret['bias_rate_without_mid'] = bias_rate_without_mid / (diffusionLoader.dataset.__len__() // args.batch)
		ret['kl_rate_without_mid'] = kl_rate_without_mid  / (diffusionLoader.dataset.__len__() // args.batch)
	
		return ret


	def testEpoch(self):
		tstLoader = self.handler.tstLoader
		epRecall, epNdcg, epPrecision = [0] * 3
		i = 0
		num = tstLoader.dataset.__len__()
		steps = num // args.tstBat

		if args.data == 'tiktok':
			usrEmbeds, itmEmbeds, side_Embeds, content_Emebeds = self.model.forward(self.handler.R, self.handler.torchBiAdj, self.image_UI_matrix, self.image_II_matrix, self.text_II_matrix, self.audio_II_matrix, None) 
		else:
			self.audio_UI_matrix, self.audio_II_matrix = None, None
			usrEmbeds, itmEmbeds, side_Embeds, content_Emebeds = self.model.forward(self.handler.R, self.handler.torchBiAdj, self.image_UI_matrix, self.image_II_matrix, self.text_II_matrix, self.audio_II_matrix, None) 

		# Inference
		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()
			allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			_, topLocs = torch.topk(allPreds, args.topk)
			recall, ndcg, precision = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
			epRecall += recall
			epNdcg += ndcg
			epPrecision += precision
			log('Steps %d/%d: recall = %.2f, ndcg = %.2f , precision = %.2f   ' % (i, steps, recall, ndcg, precision), save=False, oneline=True)
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		ret['Precision'] = epPrecision / num
		return ret

	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = allPrecision = 0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			recall = dcg = precision = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
					precision += 1
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			precision = precision / args.topk
			allRecall += recall
			allNdcg += ndcg
			allPrecision += precision
		return allRecall, allNdcg, allPrecision


def seed_it(seed):
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True 
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)

if __name__ == '__main__':
	seed_it(args.seed)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.saveDefault = True
	
	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	coach = Coach(handler)
	coach.run()