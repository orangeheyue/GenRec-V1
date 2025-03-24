# Author: Orange
# Time: 2025-03-24
# implement: GenRec-V1: Stable Interest Generation for Multi-Modal Recommendation  

import torch 
import torch.nn as  nn 
import torch.nn.functional as F

class StableInterestDiffusion():
    '''
        实现稳定的兴趣扩散过程
    '''
    def __init__(self, gamma_start, gamma_end, steps):
        super(StableInterestDiffusion, self).__init__()
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end 
        self.steps = steps
        self.alpha_bar0, self.alpha_bar1 = self.get_betas() # self.alpha_bar0：0->1累积转移概率，self.alpha_bar1 1->0累积转移概率

    def get_betas(self):
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
    
    def p_sample(self, model, x_t, t):
        '''
            反向扩散过程 
                基于贝叶斯定理计算后验转移概率
                动态平衡探索与利用，保留可靠交互的同时探索潜在兴趣
                引入pos_weight缓解交互矩阵稀疏性问题
        '''
        # 预测原始交互概率
        logits = model(x_t, t)
        x0_probs = torch.sigmoid(logits)
        
        # 计算转移权重
        if t > 0:
            # 获取前向转移概率
            prev_alpha_bar0 = self.alpha_bar0[t-1] if t>0 else 0
            prev_alpha_bar1 = self.alpha_bar1[t-1] if t>0 else 0
            
            # 计算后验分布
            p0 = x0_probs * (1 - prev_alpha_bar0) + (1 - x0_probs) * prev_alpha_bar1
            p1 = x0_probs * prev_alpha_bar0 + (1 - x0_probs) * (1 - prev_alpha_bar1)
            
            # 根据后验分布采样
            return torch.bernoulli(p1/(p0+p1))
        else:
            return torch.bernoulli(x0_probs)

    def training_losses(self, model, x_start):
        # 随机采样时间步
        t = torch.randint(0, self.steps, (x_start.size(0),))
        
        # 前向扩散
        x_t = self.q_sample(x_start, t)
        
        # 模型预测
        logits = model(x_t, t)
        
        # 二元交叉熵损失
        loss = F.binary_cross_entropy_with_logits(
            logits, x_start.float(), 
            pos_weight=torch.tensor([5.0]).cuda()  # 处理类别不平衡
        )
        
        return loss
    
    def p_mean_variance(self, model, x_t, t, modality):
        # 分模态处理特征
        if modality == 'image':
            logits = model.image_net(x_t, t)
        elif modality == 'text':
            logits = model.text_net(x_t, t)
        
        # 计算各模态转移概率
        probs = torch.sigmoid(logits)
        return {
            'probs': probs,
            'logits': logits
        }