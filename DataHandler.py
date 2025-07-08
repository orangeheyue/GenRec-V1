import pickle
import numpy as np
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
from collections import defaultdict
from tqdm import tqdm
import random

class DataHandler:
	def __init__(self):
		if args.data == 'baby':
			predir = './Datasets/baby/'
		elif args.data == 'sports':
			predir = './Datasets/sports/'
		elif args.data == 'tiktok':
			predir = './Datasets/tiktok/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'

		self.imagefile = predir + 'image_feat.npy'
		self.textfile = predir + 'text_feat.npy'
		if args.data == 'tiktok':
			self.audiofile = predir + 'audio_feat.npy'

	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
			# ret = pickle.load(fs)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat): 
		'''
			TODO: 这种预处理归一化是最优的吗？
		'''
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)
	
		# mat_ = mat.tolil()
		print("mat:", mat)
		# slice_mat = mat_[: args.user, args.user :] 
		mat_ = mat.tocsr()[: args.user, args.user :].tocoo()
		print("mat_:", mat_)
		# print("mat.tolil():", mat.tolil())
		# print('---->', slice_mat[:5])
		# make sub tensor
		idxs_ = torch.from_numpy(np.vstack([mat_.row, mat_.col]).astype(np.int64))
		vals_ = torch.from_numpy(mat_.data.astype(np.float32))
		shape_ = torch.Size(mat_.shape)
		data_ = torch.sparse.FloatTensor(idxs_, vals_, shape_).cuda()

		# print("data_:", data_)
		# make cuda tensor
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		data = torch.sparse.FloatTensor(idxs, vals, shape).cuda()
		# print("sparse data:", data[: 1, 1 :] )
		print("data:", data)
		print("data_:", data_)
		return data_, data
	


	def loadFeatures(self, filename):
		feats = np.load(filename)
		return torch.tensor(feats).float().cuda(), np.shape(feats)[1]

	def LoadData(self):
		'''
			一次性把数据加载到内存里
			trnMat.shape: (9308, 6710) :  (user ID, item ID) true interaction
				(9305, 350)   1.0
				(9306, 1076)  1.0
				(9307, 1334)  1.0
				......
			self.trnMat.A: [[0. 0. 0. ... 0. 0. 0.]
							[0. 0. 0. ... 0. 0. 0.]
							[0. 1. 0. ... 0. 0. 0.]
							...
							[0. 0. 0. ... 0. 0. 0.]
							[0. 0. 0. ... 0. 0. 0.]
							[0. 0. 0. ... 0. 0. 0.]]
			self.trnMat.A.shape: (9308, 6710)

						加载多模态特征：
			image_feats.shape: torch.Size([6710, 128])
			text_feats.shape: torch.Size([6710, 768])
			audio_feats.shape: torch.Size([6710, 128])
		'''
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		self.trnMat = trnMat
		# print("trnMat:", trnMat)
		# print("trnMat.shape:", trnMat.shape)
		# print("self.trnMat.A:", self.trnMat.A)
		# print("self.trnMat.A.shape:", (self.trnMat.A).shape)
		args.user, args.item = trnMat.shape
		self.R , self.torchBiAdj = self.makeTorchAdj(trnMat) # 生成(user+item, user+item)稀疏邻接矩阵 torchBiAdj.shape: torch.Size([16018, 16018])

		trnData = TrnData(trnMat) # 构造一个可索引的dataloader,返回user 索引, item索引, 负样本索引
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

		'''
			加载多模态特征：
			image_feats.shape: torch.Size([6710, 128])
			text_feats.shape: torch.Size([6710, 768])
			audio_feats.shape: torch.Size([6710, 128])
		'''
		self.image_feats, args.image_feat_dim = self.loadFeatures(self.imagefile)
		self.text_feats, args.text_feat_dim = self.loadFeatures(self.textfile)
		if args.data == 'tiktok':
			self.audio_feats, args.audio_feat_dim = self.loadFeatures(self.audiofile)
		else:
			self.audio_feats = None
		# print("image_feats:", self.image_feats)
		print("image_feats.shape:", self.image_feats.shape)
		# print("text_feats:", self.text_feats)
		print("text_feats.shape:", self.text_feats.shape)
		# print("audio_feats:", self.audio_feats)
		# print("audio_feats.shape:", self.audio_feats.shape)

		'''
			加载训练的User-Item交互矩阵
		
		'''
		self.diffusionData = DiffusionData(torch.FloatTensor(self.trnMat.A))
		self.diffusionLoader = dataloader.DataLoader(self.diffusionData, batch_size=args.batch, shuffle=True, num_workers=0)
		# 多模态特征
		self.multimodalFeatureData =  MultimodalFeatureDataset(self.image_feats, self.text_feats, self.audio_feats)
		self.multimodalFeatureLoader = dataloader.DataLoader(self.multimodalFeatureData, batch_size=args.batch, shuffle=False, num_workers=0) # no shuffle


class TrnData(data.Dataset):
	'''
		训练dataloader
	'''
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		'''
			negSampling 方法用于为每个正样本对应的用户随机采样一个不存在交互的负样本（物品） 
			TODO: 能否生成负样本？这里还可以挖掘一下吗【负样本采样】
		'''
		for i in range(len(self.rows)):
			u = self.rows[i]
			'''
			'''
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		'''
			user id index, item id index, 负样本采样的id index
			self.rows[idx], self.cols[idx], self.negs[idx]: 3929 1996 444
			self.rows[idx], self.cols[idx], self.negs[idx]: 47 5906 2331
		'''
		# print("self.rows[idx], self.cols[idx], self.negs[idx]:", self.rows[idx], self.cols[idx], self.negs[idx])
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	'''
		测试dataloader
	'''
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		'''
			 9307 [0. 0. 0. ... 0. 0. 0.]
		'''
		# print("self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1]):", self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1]))
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
	
class DiffusionData(data.Dataset):
	def __init__(self, data):
		self.data = data

	def __getitem__(self, index):
		item = self.data[index]
		'''
		item: tensor([0., 0., 0.,  ..., 0., 0., 0.]) index: 2219
		item: tensor([0., 0., 0.,  ..., 0., 0., 0.]) index: 4580
		item: tensor([0., 0., 0.,  ..., 0., 0., 0.]) index: 7334
		'''
		# print("item:", item, "index:", index)
		return item, index
	
	def __len__(self):
		return len(self.data)
	

class MultimodalFeatureDataset(data.Dataset):
	'''
		多模态特征dataset
	'''
	def __init__(self, image_feats, text_feats, audio_feats):
		

		self.image_feats = image_feats
		self.text_feats = text_feats
		self.audio_feats = audio_feats 

	def __len__(self):
		return self.image_feats.shape[0]
	
	def __getitem__(self, index):
		image_modal_feature = self.image_feats[index]
		text_modal_feature = self.text_feats[index]
		if self.audio_feats is not None:
			audio_modal_feature = self.audio_feats[index]

		# print("image_modal_feature.shape:",image_modal_feature.shape, "text_modal_feature.shape:", text_modal_feature.shape, "audio_modal_feature.shape:", audio_modal_feature.shape)
		return (image_modal_feature, text_modal_feature, audio_modal_feature) if self.audio_feats is not None else (image_modal_feature, text_modal_feature)
	

	