# Author: OrangeAI Research Team
# Time: 2025-03-24
# Official implementation: GenRec-V1 | Flip is Better than Noise: Unbiased Interest Generation for Multimedia Recommendation

import torch 
import torch.nn as nn 

class FlipInterestDiffusion(nn.Module):
	'''
		Implement a stable interest diffusion processArgs:steps: diffusion time stepsFunctions:
		get_cum(): obtain the cumulative transition probabilities of time steps
		q_sample(): forward diffusion process
		p_sample(): reverse diffusion process
		training_losses(): loss function
	'''
	def __init__(self, steps=5, base_temp=1.0):
		super(FlipInterestDiffusion, self).__init__()  
		self.eps = 1e-8 
		self.steps = steps
		self.base_temp = base_temp  
		#self.alpha_bar0, self.alpha_bar1 = self.get_cum() # self.alpha_bar0：0->1 cumulative transition probability state trans，self.alpha_bar1 1->0 cumulative transition probability
		
	def _compute_sparsity(self, x):
		"""
			Calculate the sparsity (proportion of zeros) of batch data
		"""
		return (x == 0).float().mean()

	def _auto_schedule_params(self, x_start):
		"""
			Automatically schedule the gamma and epsilon parameters based on input data
		"""
		sparsity = self._compute_sparsity(x_start)
		# Gamma (0->1 transition probability parameter)
		# The higher the sparsity, the larger the gamma_start (to encourage more exploration in the early stage)
		gamma_start = 0.1 * (1 - sparsity) + 0.001  #  [0.001, 0.1]
		gamma_end = gamma_start * 0.1  
		# Epsilon (1->0 transition probability parameter)
		# The higher the sparsity, the smaller the epsilon (to protect existing 1 interactions)
		epsilon_start = 0.005 * sparsity + 0.0001  #  [0.0001, 0.005]
		epsilon_end = epsilon_start * 0.1

		return gamma_start, gamma_end, epsilon_start, epsilon_end

	def get_cum(self, x_start):
		"""
			Dynamically generate data-based cumulative transition probabilities
		"""
		gamma_start, gamma_end, epsilon_start, epsilon_end = self._auto_schedule_params(x_start)
		# Gamma Scheduling (0->1 probability decreases over time)
		gamma = torch.linspace(gamma_start, gamma_end, self.steps)
		# Epsilon Scheduling (1->0 probability remains extremely low)
		epsilon = torch.linspace(epsilon_start, epsilon_end, self.steps)
		epsilon = torch.clamp(epsilon, max=0.01)  
		# Calculate the cumulative transition probability
		gamma_cum = 1 - torch.cumprod(1 - gamma, dim=0)
		epsilon_cum = 1 - torch.cumprod(1 - epsilon, dim=0)
		
		return gamma_cum.to(x_start.device), epsilon_cum.to(x_start.device)
	
	@staticmethod
	def generate_custom_noise(x, temp_scale=1.0, mode='randn'):
		"""
			Adaptive noise generation with temperature control
		"""
		if mode == 'randn':
			mean = x.float().mean()
			var = x.float().var(unbiased=True)
			std = torch.sqrt(var + 1e-8) * temp_scale 
			noise = torch.randn_like(x.float())
			return noise * std + mean
		if  mode == 'rand':
			noise = torch.rand_like(x.float())
			return noise

	
	def q_sample(self, x_start, t, temp_scale=1.0):
		"""
			x_start:
			 		[tensor([[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					...,
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.]])

			x_start.shape: torch.Size([1024, 6710])
			t: tensor([1, 3, 1,  ..., 2, 1, 2], device='cuda:0') t.shape: torch.Size([1024]) 
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
		# Dynamically generate data-based cumulative transition probabilities
		gamma_cum, epsilon_cum = self.get_cum(x_start)
		
		# Extract the cumulative probability at the current time step
		self.alpha_bar0_t = self._extract_into_tensor(gamma_cum, t, x_start.shape)
		self.alpha_bar1_t = self._extract_into_tensor(epsilon_cum, t, x_start.shape)
		# Adaptive noise generation with temperature control
		noise = self.generate_custom_noise(x_start, temp_scale, mode = 'rand')

		# Calculate the flip probability
		flip_prob = torch.where(
			x_start == 0,
			torch.sigmoid((self.alpha_bar0_t - noise) * self.base_temp),  # 0->1
			torch.sigmoid((self.alpha_bar1_t - noise) * self.base_temp)   # 1->0
		)
		
		# flip the bits based on bernoulli flip probability distribution
		flip_mask = torch.bernoulli(flip_prob)
		# print("flip_mask:", flip_mask)
		x_t = x_start.clone()
		x_t[flip_mask.bool()] = 1 - x_t[flip_mask.bool()]
		
		return x_t

	def p_sample(self, model, x_start, steps, bayesian_samplinge_schedule=True):
		'''
			model: Deniosed Model
			x_start:
			 		[tensor([[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					...,
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.],
					[0., 0., 0.,  ..., 0., 0., 0.]])

			x_start.shape: torch.Size([1024, 6710])
			Reverse diffusion process
			Calculate posterior transition probabilities based on Bayes' theorem
			Dynamically balance exploration and exploitation, exploring potential interests while retaining reliable interactions
			Introduce pos_weight to alleviate the sparsity issue of the interaction matrix

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
			t = torch.tensor([steps - 1] * batch_size).cuda() 
			x_t = self.q_sample(x_start, t) 
		# print("self.steps:", self.steps)
		indices = list(range(self.steps))[::-1]

		for i in indices:
			# assert 0 <= i < self.steps, f'Invalid step index i={i} with total steps={self.steps}'
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			logits, probs = self.p_interest_shift_probs(model, x_t, t) # torch.Size([1024, 6710])
			if bayesian_samplinge_schedule == True and i > 0:
				# Obtain the forward transition probability
				prev_alpha_bar0_t = self._extract_into_tensor(self.alpha_bar0_t, t-1, x_start.shape) # self.alpha_bar0[t-1]  torch.Size([1024, 6710])
				prev_alpha_bar1_t = self._extract_into_tensor(self.alpha_bar1_t, t-1, x_start.shape) # self.alpha_bar1[t-1]  torch.Size([1024, 6710])
				# Calculate the posterior distribution
				p0 = probs * (1 - prev_alpha_bar0_t) + (1 - probs) * prev_alpha_bar1_t
				p1 = probs * prev_alpha_bar0_t + (1 - probs) * (1 - prev_alpha_bar1_t)
				# Sample from the posterior distribution
				x_t = torch.bernoulli(p1 /(p0 + p1))
				#print("x_t:", x_t)
			else:
				x_t =  torch.bernoulli(probs)
	
		return x_t, probs

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats, text_feats, audio_feats):
		'''
			In the loss function design of diffusion models, the following issues need to be considered:
			The traditional MSE loss should be replaced with the binary cross-entropy loss.
			Since the number of un-click information (value 0) is far greater than that of interacted information (value 1), the class imbalance problem needs to be addressed.

		'''
		# Dynamic class weight calculation
		pos_weight = torch.sum(1 - x_start) / (torch.sum(x_start) + 1e-8)
		# pos_weight =  (torch.sum(x_start) + 1e-8) / torch.sum(1 - x_start)
		batch_size = x_start.size(0)
		# Randomly sample time steps
		t = torch.randint(0, self.steps, (batch_size,)).long().cuda() # t: tensor([1, 3, 1,  ..., 2, 1, 2], device='cuda:0') t.shape: torch.Size([1024]) 
		# Forward generation
		x_t = self.q_sample(x_start, t) # torch.Size([1024, 6710])
		logits, probs = self.p_interest_shift_probs(model, x_t, t)
		###########################Focal Loss###########################
		gamma = 2.0  # Focus Parameter: The larger its value, the higher the attention paid to hard samples 2.0 
		alpha = 0.25 
		p = torch.sigmoid(logits)
		p = torch.clamp(p, min=1e-7, max=1-1e-7) 
		adaptive_alpha = alpha * pos_weight.detach() 
		# print("adaptive_alpha:", adaptive_alpha)
		# Calculate the positive and negative sample masks
		pos_mask = x_start.float()
		neg_mask = 1 - pos_mask
		pos_loss = -adaptive_alpha * (1 - p).pow(gamma) * pos_mask * torch.log(p)
		neg_loss = -(1 - adaptive_alpha) * p.pow(gamma) * neg_mask * torch.log(1 - p)
		#focal_loss = (pos_loss + neg_loss).mean()
		focal_loss = (pos_loss + neg_loss).sum() / (pos_mask.sum() + neg_mask.sum() + 1e-8)
		bce_loss = F.binary_cross_entropy_with_logits(
			logits, x_start.float(), 
			pos_weight=pos_weight.cuda() 
		)
		# The contrastive loss between the modal vectors of the original graph convolution and the generated modal vectors, which brings the generated positive modal samples closer to the original ones.
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
			
		kl_loss = self._calc_kl_divergence(x_start, x_t, t, probs)
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
		post_probs = self._true_posterior(x0, xt, t).detach()
		post_probs = torch.clamp(post_probs, self.eps, 1-self.eps)
		probs = torch.clamp(probs.detach(), self.eps, 1-self.eps)
		
		kl = post_probs * (torch.log(post_probs) - torch.log(probs))
		kl += (1 - post_probs) * (torch.log(1 - post_probs) - torch.log(1 - probs))
		return kl.mean(dim=1)

	def _true_posterior(self, x0, xt, t):
		# alpha0 = self.alpha_bar0[t].view(-1,1)
		# alpha1 = self.alpha_bar1[t].view(-1,1)
		alpha0 = self._extract_into_tensor(self.alpha_bar0_t, t, x0.shape)
		alpha1 = self._extract_into_tensor(self.alpha_bar1_t, t, x0.shape)

		case0 = (x0 == 0).float() * (1 - alpha0)
		case1 = (x0 == 1).float() * alpha1
		numerator = case0 + case1

		denom0 = (x0 == 0).float() * (1 - alpha0 + alpha1)
		denom1 = (x0 == 1).float() * (alpha0 + 1 - alpha1)
		denominator = denom0 + denom1
		
		return numerator / (denominator + self.eps)
	
	def p_interest_shift_probs(self, model, x_t, t):
		logits = model(x_t, t) # model_output.shape: torch.Size([1024, 6710]
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
	