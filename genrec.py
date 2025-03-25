# Author: Orange
# Time: 2025-03-24
# implement: GenRec-V1: Stable Interest Generation for Multi-Modal Recommendation  

import torch 
import torch.nn as  nn 
import torch.nn.functional as F
import numpy as np

class StableInterestDiffusion():
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
    def __init__(self, gamma_start, gamma_end, steps):
        super(StableInterestDiffusion, self).__init__()
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end 
        self.steps = steps
        self.alpha_bar0, self.alpha_bar1 = self.get_cum() # self.alpha_bar0：0->1累积转移概率，self.alpha_bar1 1->0累积转移概率

    def get_cum(self):
        # 0->1概率线性增长，1->0概率保持极低
        gamma = torch.linspace(self.gamma_start, self.gamma_end, self.steps) 
        epsilon = torch.full(self.steps, 0.001)  # 固定为0.1%
        # 计算累积转移概率
        gamma_cum = 1 - torch.cumprod(1 - gamma, dim=0)
        epsilon_cum = 1 - torch.cumprod(1 - epsilon, dim=0)
        return gamma_cum, epsilon_cum

    def q_sample(self, x_start, t, noise=None):
        '''
            前向扩散过程
			x_start:
			 		[tensor([[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					...,
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.]])

			x_start.shape: torch.Size([1024, 6710])
            t:时间步
        '''
        # 获取当前时间步的累积转移概率
        alpha_bar0 = self.alpha_bar0[t].view(-1,1)
        alpha_bar1 = self.alpha_bar1[t].view(-1,1)
        # 生成随机掩码
        if noise is None:
            noise = torch.rand_like(x_start.float())
        # 计算状态翻转
        flip_mask = torch.where(
            x_start == 0, 
            noise < alpha_bar0,  # 0->1条件
            noise < alpha_bar1 # 1->0条件
        )
        # 应用翻转
        x_t = x_start.clone()
        x_t[flip_mask] = 1 - x_t[flip_mask]
        
        return x_t
    

    def p_sample(self, model, x_start, steps, bayesian_samplinge_schedule=False):
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
        '''
        batch_size = x_start.shape[0]
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * batch_size).cuda() # t时间矩阵
            x_t = self.q_sample(x_start, t) # 每个batch内的张量都进行前向扩散
        
        for i in list(range(self.steps))[::-1]:
            t = torch.tensor([steps - 1] * batch_size).cuda() # t时间矩阵
            logits, probs = self.p_interest_shift_probs(model, x_t, t)
            if bayesian_samplinge_schedule == True:
                # 获取前向转移概率
                prev_alpha_bar0 = self.alpha_bar0[t-1] if t > 0 else 0
                prev_alpha_bar1 = self.alpha_bar1[t-1] if t > 0 else 0
                # 计算后验分布
                p0 = probs * (1 - prev_alpha_bar0) + (1 - probs) * prev_alpha_bar1
                p1 = probs * prev_alpha_bar0 + (1 - probs) * (1 - prev_alpha_bar1)
                # 根据后验分布采样
                x_t = torch.bernoulli(p1 /(p0 + p1))
            else:
                x_t =  torch.bernoulli(probs)

        return x_t


    def training_losses(self, model, x_start):
        '''
            在扩散模型的损失函数设计中，需要考虑下面的问题:
            1. 传统的MSE损失要改为二元交叉熵损失
            2. 由于未交互的信息0值,远多余交互的信息1,因此需要考虑类别不均衡问题
            3. 
        '''
        batch_size = x_start.size(0)
        # 随机采样时间步
        t = torch.randint(0, self.steps, (batch_size,)).long().cuda()
        # 前向扩散
        x_t = self.q_sample(x_start, t)
        # 模型预测
        logits = model(x_t, t)
        # 二元交叉熵损失
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, x_start.float(), 
            pos_weight=torch.tensor([5.0]).cuda()  # 处理类别不平衡
        )
        
        return bce_loss
    
    def p_interest_shift_probs(self, model, x_t, t):
        # 分模态处理特征
        logits = model(x_t, t)
        # 计算各模态转移概率
        probs = torch.sigmoid(logits)
        return logits, probs
    
    def _timestep_embedding(self, t, dim):
        """生成时间步嵌入"""
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    