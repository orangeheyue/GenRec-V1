class StableInterestDiffusion(nn.Module):
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
        # 建议添加稳定性参数
        super(StableInterestDiffusion, self).__init__()  
        self.eps = 1e-8  # 缺失的稳定性参数
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end 
        self.steps = steps
        self.alpha_bar0, self.alpha_bar1 = self.get_cum() # self.alpha_bar0：0->1累积转移概率，self.alpha_bar1 1->0累积转移概率

    def get_cum(self):
        # 0->1概率线性增长，1->0概率保持极低
        gamma = torch.linspace(self.gamma_start, self.gamma_end, self.steps) 
        epsilon = torch.full((self.steps, ), 0.001)  # 固定为0.1%
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

        '''
        # 获取当前时间步的累积转移概率
        alpha_bar0_t = self._extract_into_tensor(self.alpha_bar0, t, x_start.shape)
        alpha_bar1_t = self._extract_into_tensor(self.alpha_bar1, t, x_start.shape)
        # 生成随机掩码
        if noise is None:
            noise = torch.rand_like(x_start.float())
        # 计算状态翻转
        flip_mask = torch.where(
            x_start == 0, 
            noise < alpha_bar0_t,  # 0->1条件
            noise < alpha_bar1_t # 1->0条件
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
        # 反向过程迭代
        for i in list(range(self.steps))[::-1]:
            t = torch.tensor([i] * batch_size).cuda() # t时间矩阵 t: tensor([4, 4, 4,  ..., 4, 4, 4], device='cuda:0') t.shape: torch.Size([1024])
            logits, probs = self.p_interest_shift_probs(model, x_t, t) # torch.Size([1024, 6710])
            if bayesian_samplinge_schedule == True and i > 0:
                # 获取前向转移概率
                prev_alpha_bar0 = self._extract_into_tensor(self.alpha_bar0, t-1, x_start.shape) # self.alpha_bar0[t-1]  torch.Size([1024, 6710])
                prev_alpha_bar1 = self._extract_into_tensor(self.alpha_bar1, t-1, x_start.shape) # self.alpha_bar1[t-1]  torch.Size([1024, 6710])
                # 计算后验分布
                p0 = probs * (1 - prev_alpha_bar0) + (1 - probs) * prev_alpha_bar1
                p1 = probs * prev_alpha_bar0 + (1 - probs) * (1 - prev_alpha_bar1)
                # 根据后验分布采样
                x_t = torch.bernoulli(p1 /(p0 + p1))
            else:
                x_t =  torch.bernoulli(probs)

        return x_t


    def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
        '''
            在扩散模型的损失函数设计中，需要考虑下面的问题:
            1. 传统的MSE损失要改为二元交叉熵损失
            2. 由于未交互的信息0值,远多余交互的信息1,因此需要考虑类别不均衡问题
            3. 
        '''
        batch_size = x_start.size(0)
        # 随机采样时间步
        t = torch.randint(0, self.steps, (batch_size,)).long().cuda() # t: tensor([1, 3, 1,  ..., 2, 1, 2], device='cuda:0') t.shape: torch.Size([1024]) 
        # 前向扩散
        x_t = self.q_sample(x_start, t) # torch.Size([1024, 6710])
        # 模型预测
        logits, probs = self.p_interest_shift_probs(model, x_t, t)
        # 二元交叉熵损失
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, x_start.float(), 
            pos_weight=torch.tensor([5.0]).cuda()  # 处理类别不平衡
        )
		#    # KL散度项
		#     kl_loss = self._calc_kl_divergence(x_start, x_t, t, probs)
		#     # 课程学习权重
		#     curriculum_weight = torch.clamp(t.float() / self.steps, 0, 0.5)
		#     total_loss = bce_loss + curriculum_weight * kl_loss
        
        return bce_loss
	

	# def mean_flat(self, tensor):
	# 	return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	# def SNR(self, t):
	# 	self.alphas_cumprod = self.alphas_cumprod.cuda()
	# 	return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    # def _calc_kl_divergence(self, x0, xt, t, probs):
    #     """稳健的KL散度计算"""
    #     # 真实后验概率
    #     post_probs = self._true_posterior(x0, xt, t)
        
    #     # 数值稳定性处理
    #     post_probs = torch.clamp(post_probs, self.eps, 1-self.eps)
    #     probs = torch.clamp(probs, self.eps, 1-self.eps)
        
    #     kl = post_probs * (torch.log(post_probs) - torch.log(probs))
    #     kl += (1 - post_probs) * (torch.log(1 - post_probs) - torch.log(1 - probs))
    #     return kl.mean()

    # def _true_posterior(self, x0, xt, t):
    #     """精确后验计算"""
    #     # alpha0 = self.alpha_bar0[t].view(-1,1)
    #     # alpha1 = self.alpha_bar1[t].view(-1,1)
        
	# 	alpha0 = self._extract_into_tensor(self.alpha_bar0, t, x0.shape)
	# 	alpha1 = self._extract_into_tensor(self.alpha_bar1, t, x0.shape)
    #     # 分子项
    #     case0 = (x0 == 0).float() * (1 - alpha0)
    #     case1 = (x0 == 1).float() * alpha1
    #     numerator = case0 + case1
        
    #     # 分母项
    #     denom0 = (x0 == 0).float() * (1 - alpha0 + alpha1)
    #     denom1 = (x0 == 1).float() * (alpha0 + 1 - alpha1)
    #     denominator = denom0 + denom1
        
    #     return numerator / (denominator + self.eps)
    
    def p_interest_shift_probs(self, model, x_t, t):
        # 分模态处理特征
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