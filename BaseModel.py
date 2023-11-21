import torch 
import torch.nn as nn
import torch.nn.functional as F

class GraphRec(nn.Module):
    def __init__(self, n_users, n_items, hidden, hop, drop):
        super().__init__()
        self.hop = 1
        self.hidden = hidden
        self.social_weight_dict = self.init_weight()
        self.item_weight_dict = self.init_weight()
        self.dropout = drop
        self.tanh = nn.Tanh()
        self.n_users = n_users
        self.n_items = n_items
        
    def init_weight(self):
        initializer = nn.init.xavier_uniform_
        weight_dict = nn.ParameterDict()
        for k in range(self.hop):
            weight_dict.update({'W_%d'%k: nn.Parameter(initializer(torch.empty(self.hidden,
                                                                      self.hidden)))})
        return weight_dict
    
    def GCN_a(self, users_emb, items_emb, adj):
        num_users = users_emb.size()[0]
        num_items = items_emb.size()[0]
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for k in range(self.hop):
            all_emb = torch.sparse.mm(adj, all_emb) # sparse x sparse -> sparse sparse x dense -> dense
            all_emb = self.tanh(torch.matmul(all_emb, self.item_weight_dict['W_%d' %k]))
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [num_users, num_items])
        return users, items
    
    def GCN_s(self, user_embs, adj):
        adj = F.dropout(adj, p=self.dropout, training=self.training) # 必须training=self.training 只有nn.Dropout不需要这样显式声明
        for k in range(self.hop):
            new_user_embs = torch.matmul(adj, user_embs)
            user_embs = self.tanh(torch.matmul(new_user_embs, self.social_weight_dict['W_%d' %k])) + user_embs
        return user_embs
    
    def forward(self, users, pos, neg, user_embs, item_embs, S, A):
        all_user_embs_S = self.GCN_s(user_embs, S) 
        all_user_embs_A, all_item_embs = self.GCN_a(user_embs, item_embs, A) 
        all_user_embs = 0.5*all_user_embs_S + 0.5*all_user_embs_A 
        
        users_emb = all_user_embs[users]
        pos_emb = all_item_embs[pos]
        neg_emb = all_item_embs[neg] # 经过print size 证实是[B,neg,H]
        # x[mask] 当mask为LongTensor https://blog.csdn.net/goodxin_ie/article/details/89672700 就跟nn.Embedding(input) 类似   
        return users_emb, pos_emb, neg_emb

    def rec_loss(self, user_emb, pos_emb, neg_emb):
        pos_scores = torch.sum(user_emb*pos_emb, dim=1)  # [B]
        neg_scores = torch.sum(user_emb*neg_emb, dim=1)  # [B]
        sup_logits = pos_scores-neg_scores
        bpr_loss = -torch.mean(F.logsigmoid(sup_logits))
        return bpr_loss

    def l2_loss(self, *weights):
        loss = 0.0
        for w in weights:
            loss += torch.mean(torch.pow(w, 2))

        return 0.5*loss
    
    def reg_loss(self, users, pos, neg, user_embs, item_embs):
        reg_loss = self.l2_loss(
            user_embs[users],
            item_embs[pos],
            item_embs[neg],
        )
        return reg_loss

class DESIGN(nn.Module):
    def __init__(self, n_users, n_items, hidden, hop, drop):
        super().__init__()
        self.hop = hop
        self.hidden = hidden
        self.n_users = n_users
        self.n_items = n_items
        self.user1_embs = nn.Embedding(self.n_users, self.hidden)
        self.item1_embs = nn.Embedding(self.n_items, self.hidden)
        self.user2_embs = nn.Embedding(self.n_users, self.hidden)
        self.item2_embs = nn.Embedding(self.n_items, self.hidden)
        nn.init.xavier_uniform_(self.user1_embs.weight)
        nn.init.xavier_uniform_(self.item1_embs.weight)
        nn.init.xavier_uniform_(self.user2_embs.weight)
        nn.init.xavier_uniform_(self.item2_embs.weight)
        
    def GCN_a(self, users_emb, items_emb, adj):
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for _ in range(self.hop):
            all_emb = torch.sparse.mm(adj, all_emb) # sparse x sparse -> sparse sparse x dense -> dense
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items
    
    def GCN_s(self, users_emb, adj):
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
    
    def forward(self, users, pos, neg, user_embs, item_embs, S, A):
        all_user_embs_S = self.GCN_s(user_embs, S) 
        all_user_embs_A, all_item_embs = self.GCN_a(user_embs, item_embs, A) 
        all_user_embs = 0.5*all_user_embs_S + 0.5*all_user_embs_A # twinGCN
        
        all_user_embs_social = self.GCN_s(self.user1_embs.weight, S) # socialGCN
        all_item_embs_social = self.item1_embs.weight
        
        all_user_embs_rating, all_item_embs_rating = self.GCN_a(self.user2_embs.weight, self.item2_embs.weight, A) # ratingGCN
        
        users_emb = all_user_embs[users]
        pos_emb = all_item_embs[pos]
        neg_emb = all_item_embs[neg] # 经过print size 证实是[B,neg,H]
        # x[mask] 当mask为LongTensor https://blog.csdn.net/goodxin_ie/article/details/89672700 就跟nn.Embedding(input) 类似
        
        users_emb_social = all_user_embs_social[users]
        pos_emb_social = all_item_embs_social[pos]
        neg_emb_social = all_item_embs_social[neg]

        users_emb_rating = all_user_embs_rating[users]
        pos_emb_rating = all_item_embs_rating[pos]
        neg_emb_rating = all_item_embs_rating[neg]
        
        return users_emb, pos_emb, neg_emb, users_emb_social, pos_emb_social, neg_emb_social, users_emb_rating, pos_emb_rating, neg_emb_rating

    def rec_loss(self, user_emb, pos_emb, neg_emb):
        pos_scores = torch.sum(user_emb*pos_emb, dim=1)  # [B]
        neg_scores = torch.sum(user_emb*neg_emb, dim=1)  # [B]
        sup_logits = pos_scores-neg_scores
        bpr_loss = -torch.mean(F.logsigmoid(sup_logits))
        return bpr_loss

    def l2_loss(self, *weights):
        loss = 0.0
        for w in weights:
            loss += torch.mean(torch.pow(w, 2))

        return 0.5*loss
    
    def reg_loss(self, users, pos, neg, user_embs, item_embs):
        reg_loss = self.l2_loss(
            user_embs[users],
            item_embs[pos],
            item_embs[neg],
        )
        reg_loss_social = self.l2_loss(
            self.user1_embs(users),
            self.item1_embs(pos),
            self.item1_embs(neg),
        )
        reg_loss_rating = self.l2_loss(
            self.user2_embs(users),
            self.item2_embs(pos),
            self.item2_embs(neg),
        )
        return reg_loss+reg_loss_social+reg_loss_rating

    def compute_distill_loss(self, pre_a, pre_b):
        pre_a = self.sigmoid(pre_a)
        pre_b = self.sigmoid(pre_b)
        distill_loss = - torch.mean(pre_b * torch.log(pre_a) + (1 - pre_b) * torch.log(1 - pre_a))
        return distill_loss   
    
    def KD_loss(self, users_emb, pos_emb, neg_emb, users_emb_social, pos_emb_social, neg_emb_social, users_emb_rating, pos_emb_rating, neg_emb_rating):

        pos_emb = pos_emb.unsqueeze(1) # [B,1,H]
        neg_emb = neg_emb.unsqueeze(1) # [B,1,H]
        all_item_embs = torch.cat([pos_emb, neg_emb], dim=1) # [B,2,H] cat方法按输入顺序拼接:先pos_emb, 再negs_emb
        users_emb = users_emb.unsqueeze(1) # [B,1,H]
        pre = torch.mul(users_emb, all_item_embs) # [B,1,H]*[B,2,H]=[B,2,H]
        pre = torch.mean(pre, dim=-1) # [B,2]
        
        pos_emb_social = pos_emb_social.unsqueeze(1)
        neg_emb_social = neg_emb_social.unsqueeze(1)
        all_item_embs_social = torch.cat([pos_emb_social, neg_emb_social], dim=1) 
        users_emb_social = users_emb_social.unsqueeze(1)
        pre_social = torch.mul(users_emb_social, all_item_embs_social) # [B,1,H]*[B,2,H]=[B,2,H]
        pre_social = torch.mean(pre_social, dim=-1) # [B,2]
        
        pos_emb_rating = pos_emb_rating.unsqueeze(1)
        neg_emb_rating = neg_emb_rating.unsqueeze(1)
        all_item_embs_rating = torch.cat([pos_emb_rating, neg_emb_rating], dim=1) 
        users_emb_rating = users_emb_rating.unsqueeze(1)
        pre_rating = torch.mul(users_emb_rating, all_item_embs_rating) # [B,1,H]*[B,2,H]=[B,2,H]
        pre_rating = torch.mean(pre_rating, dim=-1) # [B,2]
        kd_loss = 0
        kd_loss += self.compute_distill_loss(pre, pre_social)
        kd_loss += self.compute_distill_loss(pre, pre_rating)
        kd_loss += self.compute_distill_loss(pre_social, pre)
        kd_loss += self.compute_distill_loss(pre_social, pre_rating)
        kd_loss += self.compute_distill_loss(pre_rating, pre)
        kd_loss += self.compute_distill_loss(pre_rating, pre_social)
        return kd_loss