import torch
import torch as t
import torch.nn.functional as F

def innerProduct(usrEmbeds, itmEmbeds):
	return t.sum(usrEmbeds * itmEmbeds, dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret

def calcReward(bprLossDiff, keepRate):
	_, posLocs = t.topk(bprLossDiff, int(bprLossDiff.shape[0] * (1 - keepRate)))
	reward = t.zeros_like(bprLossDiff).cuda()
	reward[posLocs] = 1.0
	return reward

def calcGradNorm(model):
	ret = 0
	for p in model.parameters():
		if p.grad is not None:
			ret += p.grad.data.norm(2).square()
	ret = (ret ** 0.5)
	ret.detach()
	return ret

def contrastLoss(embeds1, embeds2, nodes, temp):
	embeds1 = F.normalize(embeds1, p=2)
	embeds2 = F.normalize(embeds2, p=2)
	pckEmbeds1 = embeds1[nodes]
	pckEmbeds2 = embeds2[nodes]
	nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
	deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)
	return -t.log(nume / deno).mean()

def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
    from torch_scatter import scatter_add
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight

def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm

def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        row = [i[0] for i in tuple_list]
        col = [i[1] for i in tuple_list]
        i = torch.LongTensor([row, col]).to(device)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)