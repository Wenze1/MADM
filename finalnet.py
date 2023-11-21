import torch
import torch.nn as nn
import numpy as np
from GraphLearner import GraphLearner
import scipy.sparse as sp
import torch.nn.functional as F
import random
from BaseModel import DiffNet_plus, DESIGN, GraphRec, SEPT

class MyModel(nn.Module):
    
    def __init__(self, config, args, device):
        super(MyModel, self).__init__()
        self.device = device
        self.n_users = config['n_users']
        self.n_target_users = config['n_target_users']
        self.n_items = config['n_items']
        
        # Graph
        S = config['S']
        self.S = self.convert_numpy_to_tensor(S)
        A = config['A_tr']
        self.A = self.sparse_mx_to_torch_sparse_tensor(A)
        self.RRT = config['RRT_tr']
        self.RRTdrop = None
        
        # training hyper-parameter
        self.hidden = args.hidden
        self.hop = args.hop
        self.drop = args.dropout
        self.decay = args.decay
        self.kd_reg = args.kd_reg
        self.cl_reg = args.cl_reg
        self.recon_reg = args.recon_reg
        self.recon_drop = args.recon_drop
        
        # layer
        self.user_embs = nn.Embedding(self.n_users, self.hidden) 
        self.item_embs = nn.Embedding(self.n_items, self.hidden)
        nn.init.xavier_uniform_(self.user_embs.weight)
        nn.init.xavier_uniform_(self.item_embs.weight)
        
        self.graphlearner = GraphLearner(args, self.device) 
        self.base_model_name = args.base_model_name 
        if self.base_model_name == 'DiffNet++':
            self.basemodel = DiffNet_plus(self.n_users, self.n_items, self.hidden, self.hop, self.drop)
        if self.base_model_name == 'GraphRec':
            self.basemodel = GraphRec(self.n_users, self.n_items, self.hidden, self.hop, self.drop)
        if self.base_model_name == 'SEPT':
            self.basemodel = SEPT(self.n_users, self.n_items, self.hidden, self.hop, self.drop)
        if self.base_model_name == 'DESIGN':
            self.basemodel = DESIGN(self.n_users, self.n_items, self.hidden, self.hop, self.drop)
    
    def convert_numpy_to_tensor(self, adj):
        adj = torch.FloatTensor(adj).to(self.device)
        return adj

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        Graph = torch.sparse.FloatTensor(indices, values, shape)
        Graph = Graph.coalesce().to(self.device)
        return Graph
    
    def edge_dropout(self, sp_adj):
        """Input: a sparse user-item adjacency matrix and a dropout rate."""
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        keep_idx = random.sample(range(edge_count), int(edge_count * (1 - self.recon_drop)))
        user1_np = np.array(row_idx)[keep_idx]
        user2_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(user1_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user1_np, user2_np)), shape=(self.n_users, self.n_users))
        return dropped_adj


    def calculate_loss(self, users, pos, neg, epoch):
        """
        Only this function appears in train()
        """
        users = users.long()
        pos = pos.long()
        neg = neg.long()

        new_S = self.graphlearner(self.user_embs.weight, self.S)
        # all_user_embs_1 represents minimal and sufficient user social representation
        all_user_embs_1, all_user_embs_2 = self.graphlearner.GCN(self.user_embs.weight, self.item_embs.weight, new_S, self.A)
        cl_loss = self.graphlearner.cl_loss(users, all_user_embs_1, all_user_embs_2)
        # recon_loss = self.graphlearner.reconstru_loss(all_user_embs_1, self.RRTdrop)
        reg_loss = self.basemodel.reg_loss(users, pos, neg, self.user_embs.weight, self.item_embs.weight)
        if self.base_model_name == 'DiffNet++' or self.base_model_name == 'GraphRec' or self.base_model_name == 'SEPT':
            users_emb, pos_emb, neg_emb = self.basemodel(users, pos, neg, self.user_embs.weight, self.item_embs.weight, new_S, self.A)
            bpr_loss = self.basemodel.rec_loss(users_emb, pos_emb, neg_emb)
        if self.base_model_name == 'DESIGN':
            users_emb, pos_emb, neg_emb, users_emb_social, pos_emb_social,\
                neg_emb_social, users_emb_rating, pos_emb_rating, neg_emb_rating = self.basemodel(users, pos, neg, self.user_embs.weight, self.item_embs.weight, new_S, self.A)
            kd_loss = self.basemodel.KD_loss(users_emb, pos_emb, neg_emb, users_emb_social, pos_emb_social,\
                neg_emb_social, users_emb_rating, pos_emb_rating, neg_emb_rating)
            bpr_loss = self.basemodel.rec_loss(users_emb, pos_emb, neg_emb)+ self.basemodel.rec_loss(users_emb_social, pos_emb_social,neg_emb_social)+\
                self.basemodel.rec_loss(users_emb_rating, pos_emb_rating, neg_emb_rating)
        
        if self.base_model_name == 'DiffNet++' or self.base_model_name == 'GraphRec' or self.base_model_name == 'SEPT':
            loss = bpr_loss+self.decay*reg_loss+ self.cl_reg*cl_loss
            # loss = bpr_loss+self.decay*reg_loss+  self.recon_reg*recon_loss
        if self.base_model_name == 'DESIGN':
            loss = bpr_loss+self.decay*reg_loss + self.kd_reg*kd_loss+self.cl_reg*cl_loss
            # loss = bpr_loss+self.decay*reg_loss+ self.kd_reg*kl_loss+  self.recon_reg*recon_loss
        return loss
    
    def batch_full_sort_predict(self, users, pos, neg_items, epoch):
        users = users.long()
        pos = pos.long()
        neg_items = neg_items.long()

        new_S = self.graphlearner(self.user_embs.weight, self.S)
        if self.base_model_name == 'DiffNet++' or self.base_model_name == 'GraphRec' or self.base_model_name == 'SEPT':
            user_emb, pos_emb, negs_emb = self.basemodel(users, pos, neg_items, self.user_embs.weight, self.item_embs.weight, new_S, self.A)
        if self.base_model_name == 'DESIGN':
            user_emb, pos_emb, negs_emb, _, _, _, _, _, _ = self.basemodel(users, pos, neg_items, self.user_embs.weight, self.item_embs.weight, new_S, self.A)
        pos_emb = pos_emb.unsqueeze(1) # [B,1,H]
        all_item_embs = torch.cat([pos_emb, negs_emb], dim=1) # [B,N=1+999,H] cat方法按输入顺序拼接:先pos_emb, 再negs_emb
        user_emb = user_emb.unsqueeze(1) # [B,1,H]
        scores = torch.mul(user_emb, all_item_embs) # [B,1,H]*[B,N,H]=[B,N,H]
        scores = torch.mean(scores, dim=-1) # [B,N] 

        scores, indices = torch.sort(scores, dim=-1, descending=True) # torch.sort https://hxhen.com/archives/226
        rank = torch.argwhere(indices==0)[:,1] # [B] rank最小为0 np.where https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
        isTop10 = (rank<10)
        isTop5 = (rank<5)
        isTop15 = (rank<15)
        logrank =  1/torch.log2(rank+2)
        
        HT5 = torch.sum(isTop5)
        HT10 = torch.sum(isTop10)
        HT15 = torch.sum(isTop15)
        NDCG5 = torch.sum(isTop5*logrank)
        NDCG10 = torch.sum(isTop10*logrank)
        NDCG15 = torch.sum(isTop15*logrank)
        return HT5, HT10, HT15, NDCG5, NDCG10, NDCG15

