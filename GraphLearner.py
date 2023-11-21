import torch 
import torch.nn as nn
import torch.nn.functional as F

class LightGCN_s(nn.Module):
    def __init__(self, hop):
        super(LightGCN_s, self).__init__()
        self.hop = hop
    
    def forward(self, users_emb, adj):
        embs = [users_emb]
        for _ in range(self.hop):
            if adj.is_sparse:
                users_emb = torch.sparse.mm(adj, users_emb) # sparse x sparse -> sparse sparse x dense -> dense
            else:
                users_emb = torch.matmul(adj, users_emb)
            embs.append(users_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        return light_out

class LightGCN_a(nn.Module):
    def __init__(self, hop):
        super(LightGCN_a, self).__init__()
        self.hop = hop

    def forward(self, users_emb, items_emb, adj):
        num_users = users_emb.size()[0]
        num_items = items_emb.size()[0]
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for _ in range(self.hop):
            all_emb = torch.sparse.mm(adj, all_emb) # sparse x sparse -> sparse sparse x dense -> dense
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [num_users, num_items])
        return users

class GraphLearner(nn.Module):
    
    def __init__(self, args, device):
        super().__init__()
        self.input_size = args.hidden # cosine similarity size
        self.topk = args.graph_learn_top_k_S
        self.epsilon = args.graph_learn_epsilon
        self.num_pers = args.graph_learn_num_pers
        self.device = device
        self.metric_type = args.metric_type
        self.graph_skip_conn = args.graph_skip_conn
        self.cl_temp = args.cl_temp
        
        if self.metric_type == 'weighted_cosine':
            self.weight_tensor = torch.Tensor(self.num_pers, self.input_size)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
        
        self.lightgcn_s = LightGCN_s(hop=args.graph_learner_hop)
        self.lightgcn_a = LightGCN_a(hop=args.graph_learner_hop)
        
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss() # default: mean
        self.mse = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        
        
    def forward(self, feature, init_graph):
        if self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1) # [4, 1, feat_dim]
            # print("self.weight_tensor.size(): ", self.weight_tensor.size())
            context_fc = feature.unsqueeze(0) * expand_weight_tensor # [1, N, feat_dim]*[4, 1, feat_dim]=[4, N, feat_dim]
            # cosine_similarity(a,b)={a/|a|^2}*{b/|b|^2}
            context_norm = F.normalize(context_fc, p=2, dim=-1) # 指定维度上做均一化 v/|v|^2 
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0) # BB^T=[4, N, feat_dim][4, feat_dim, N]->[N, N]求similarity
            maskoff_value = 0   
        # graph structure learning 更新出的图本质上就是attention 每个位置上是对应权重
        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, maskoff_value)

        # if self.topk is not None:
        #     attention = self.build_knn_neighbourhood(attention, self.topk, maskoff_value)
        
        assert attention.min().item() >= 0
        
        # print(attention.size()) # [7317, 7317]
        
        # learned_graph = normalize_dense(attention)
        learned_graph = attention / torch.clamp(torch.sum(attention, dim=-1, keepdim=True), min=1e-12) # row-normalization 
        learned_graph = self.graph_skip_conn * init_graph + (1-self.graph_skip_conn) * learned_graph
        # graph_skip_conn = 1-min((epoch+1)/500, 1)
        # learned_graph = graph_skip_conn * init_graph + (1-graph_skip_conn) * learned_graph
        
        return learned_graph

    def build_epsilon_neighbourhood(self, attention, epsilon, maskoff_value):
        """
        Tensor.detach() is used to detach a tensor from the current computational graph. It returns a new tensor that doesn't require a gradient.

            When we don't need a tensor to be traced for the gradient computation, we detach the tensor from the current computational graph.

            We also need to detach a tensor when we need to move the tensor from GPU to CPU.
        """
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + maskoff_value * (1 - mask)
        return weighted_adjacency_matrix

    def build_knn_neighbourhood(self, attention, topk, maskoff_value):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = (maskoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val).to(self.device)
        return weighted_adjacency_matrix
    
    def GCN(self, users_emb, items_emb, S, A):
        all_user_embs_1 = self.lightgcn_s(users_emb, S)
        all_user_embs_2 = self.lightgcn_a(users_emb, items_emb, A)
        return all_user_embs_1, all_user_embs_2
    
    def cl_loss(self, users, all_user_embs_1, all_user_embs_2):
        all_user_embs_1 = F.normalize(all_user_embs_1, dim=1)
        all_user_embs_2 = F.normalize(all_user_embs_2, dim=1)
        user_1_embs = all_user_embs_1[users]
        user_2_embs = all_user_embs_2[users]
        pos_ratings_user = torch.sum(
            user_1_embs*user_2_embs, dim=-1)  # [B] 分子

        tot_rating_user = torch.matmul(user_1_embs,
                                       torch.transpose(all_user_embs_2, 0, 1))  # [B,d]*[d, N]=[B，N] 分母 总样本是N，负样本是N-1
        pos_ratings_user = pos_ratings_user.unsqueeze(1)
        cl_logits_user = tot_rating_user - \
            pos_ratings_user  # 分母-分子 [B,N]-[B,1]=[B,N]
        clogits_user = torch.logsumexp(cl_logits_user / self.cl_temp, dim=1)
        infonce_loss = torch.mean(clogits_user)
        return infonce_loss
    
    def reconstru_loss(self, all_user_embs, RRT):
        recon = torch.matmul(all_user_embs, torch.transpose(all_user_embs, 0, 1)) # [N, N]
        # recon_loss = self.bce(self.sigmoid(recon), RRT)
        recon_loss = self.mse(self.sigmoid(recon), RRT)
        # recon_loss = self.l1loss(self.sigmoid(recon), RRT)
        return recon_loss