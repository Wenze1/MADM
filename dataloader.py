import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import scipy.sparse as sp
import torch
from utils import normalize_dense, normalize_sp

def get_spRRT(config, args, rating_valid, rating_test):
    n_users = config['n_users']
    n_items = config['n_items']
    all_ratings = config['user_rating']
    dir5 = f'../raw dataset/{args.dataset}/sp_uu_collab_adj_mat_tr0613.npz'
    dir6 = f'../raw dataset/{args.dataset}/sp_uu_collab_adj_mat_val0613.npz'
    try:
        sp_uu_collab_adj_mat_tr0613 = sp.load_npz(dir5)
        sp_uu_collab_adj_mat_val0613 = sp.load_npz(dir6)
        print("already load sparse RRT")
    except Exception:
        sp_uu_collab_adj_mat_tr0613, sp_uu_collab_adj_mat_val0613 = create_spRRT(n_users, n_items, all_ratings, rating_valid, rating_test)
        sp.save_npz(dir5, sp_uu_collab_adj_mat_tr0613)
        sp.save_npz(dir6, sp_uu_collab_adj_mat_val0613)
    return sp_uu_collab_adj_mat_tr0613, sp_uu_collab_adj_mat_val0613

def create_spRRT(n_users, n_items, all_ratings, rating_valid, rating_test):
    R_tr = np.zeros((n_users, n_items))
    R_val = np.zeros((n_users, n_items))
    for uid in all_ratings.keys():
        for item in all_ratings[uid]:
            if item not in rating_test[uid]: # 不是 not item == rating_test[uid] 因为我们的dict values是list
                R_val[uid-1, item-n_users-1] = 1
                if item not in rating_valid[uid]:
                    R_tr[uid-1, item-n_users-1] = 1 # idx convert uid:0---n_users-1 itemid:0---n_items-1-n_users 这样可以两个nn.Embedding(n_users) (n_items)
    uu_collab_adj_mat_tr = np.dot(R_tr, R_tr.T) # 得到的矩阵对角元素巨大ri*ri 其他位置稀疏 options:1.对角元素处理 2.每个item至少n个交互才能稠密
    row, col = np.diag_indices_from(uu_collab_adj_mat_tr)
    uu_collab_adj_mat_tr[row, col] = 1
    uu_collab_adj_mat_tr = sp.dok_matrix(uu_collab_adj_mat_tr)

    uu_collab_adj_mat_val = np.dot(R_val, R_val.T)
    uu_collab_adj_mat_val[row, col] = 1
    uu_collab_adj_mat_val = sp.dok_matrix(uu_collab_adj_mat_val)
    
    return uu_collab_adj_mat_tr.tocsr(), uu_collab_adj_mat_val.tocsr()

def get_RRT(config, args, rating_valid, rating_test):
    n_users = config['n_users']
    n_items = config['n_items']
    all_ratings = config['user_rating']
    dir5 = f'../raw dataset/{args.dataset}/uu_collab_adj_mat0613.npz'
    try:
        uu_collab_adj_mat0613 = np.load(dir5)
        uu_collab_adj_mat_tr0613 = uu_collab_adj_mat0613['tr']
        uu_collab_adj_mat_val0613 = uu_collab_adj_mat0613['val']
        print("already load RRT")
    except Exception:
        uu_collab_adj_mat_tr0613, uu_collab_adj_mat_val0613 = create_RRT(n_users, n_items, all_ratings, rating_valid, rating_test)
        np.savez(dir5, tr=uu_collab_adj_mat_tr0613, val=uu_collab_adj_mat_val0613)
    return uu_collab_adj_mat_tr0613, uu_collab_adj_mat_val0613

def create_RRT(n_users, n_items, all_ratings, rating_valid, rating_test):
    R_tr = np.zeros((n_users, n_items))
    R_val = np.zeros((n_users, n_items))
    for uid in all_ratings.keys():
        for item in all_ratings[uid]:
            if item not in rating_test[uid]: # 不是 not item == rating_test[uid] 因为我们的dict values是list
                R_val[uid-1, item-n_users-1] = 1
                if item not in rating_valid[uid]:
                    R_tr[uid-1, item-n_users-1] = 1 # idx convert uid:0---n_users-1 itemid:0---n_items-1-n_users 这样可以两个nn.Embedding(n_users) (n_items)
    uu_collab_adj_mat_tr = np.dot(R_tr, R_tr.T) # 得到的矩阵对角元素巨大ri*ri 其他位置稀疏 options:1.对角元素处理 2.每个item至少n个交互才能稠密
    row, col = np.diag_indices_from(uu_collab_adj_mat_tr)
    uu_collab_adj_mat_tr[row, col] = 1
    uu_collab_adj_mat_tr = normalize_dense(uu_collab_adj_mat_tr)

    uu_collab_adj_mat_val = np.dot(R_val, R_val.T)
    uu_collab_adj_mat_val[row, col] = 1
    uu_collab_adj_mat_val = normalize_dense(uu_collab_adj_mat_val)
    
    return uu_collab_adj_mat_tr, uu_collab_adj_mat_val
    

def get_adj_mat(config, args, rating_valid, rating_test):
    n_users = config['n_users']
    n_target_users = config['n_target_users']
    n_items = config['n_items']
    social_network = config['user_social']
    all_ratings = config['user_rating']
    dir1 = f'../raw dataset/{args.dataset}/uu_collab_adj_mat.npz'
    dir2 = f'../raw dataset/{args.dataset}/uu_social_adj_mat.npz'
    dir3 = f'../raw dataset/{args.dataset}/adj_mat_tr.npz'
    dir4 = f'../raw dataset/{args.dataset}/adj_mat_val.npz'
    try:
        uu_collab_adj_mat = np.load(dir1)
        uu_social_adj_mat = np.load(dir2)
        uu_collab_adj_mat_tr = uu_collab_adj_mat['tr']
        uu_collab_adj_mat_val = uu_collab_adj_mat['val']        
        uu_social_adj_mat = uu_social_adj_mat['all']
        A_tr = sp.load_npz(dir3)
        A_val = sp.load_npz(dir4)
        print('already load')
    
    except Exception:
        uu_collab_adj_mat_tr, uu_collab_adj_mat_val,  uu_social_adj_mat, A_tr, A_val \
            = create_adj_mat(n_users, n_items, all_ratings, rating_valid, rating_test, social_network)
        np.savez(dir1, tr=uu_collab_adj_mat_tr, val=uu_collab_adj_mat_val)
        np.savez(dir2, all=uu_social_adj_mat)
        sp.save_npz(dir3, A_tr)
        sp.save_npz(dir4, A_val)
    return uu_collab_adj_mat_tr, uu_collab_adj_mat_val,  uu_social_adj_mat, A_tr, A_val

def create_adj_mat(n_users, n_items, all_ratings, rating_valid, rating_test, social_network):
    """
      预处理的数据中item_idx += n_users (item_idx from 1)
      user中只有rating无social的部分已经被去除 所以all_ratings里存下的user全都是1--n_target_users  user_idx:1--n_target_users, n_target_users--n_users
      social_adj: [n_users, n_users]    collab_adj: [n_users, n_users]  # collab必须设置为n_users  n_target_users---n_users部分的RRT为0就好了 不影响其他用户
      上述情况在构建邻接矩阵的时候都要考虑
    """
    R_tr = np.zeros((n_users, n_items))
    R_val = np.zeros((n_users, n_items))
    for uid in all_ratings.keys():
        for item in all_ratings[uid]:
            if item not in rating_test[uid]: # 不是 not item == rating_test[uid] 因为我们的dict values是list
                R_val[uid-1, item-n_users-1] = 1
                if item not in rating_valid[uid]:
                    R_tr[uid-1, item-n_users-1] = 1 # idx convert uid:0---n_users-1 itemid:0---n_items-1-n_users 这样可以两个nn.Embedding(n_users) (n_items)
    uu_collab_adj_mat_tr = np.dot(R_tr, R_tr.T) # 得到的矩阵对角元素巨大ri*ri 其他位置稀疏 options:1.对角元素处理 2.每个item至少n个交互才能稠密
    uu_collab_adj_mat_tr = normalize_dense(uu_collab_adj_mat_tr)
    uu_collab_adj_mat_val = np.dot(R_val, R_val.T)
    uu_collab_adj_mat_val = normalize_dense(uu_collab_adj_mat_val)
    
    S = np.zeros((n_users, n_users)) 
    for uid in social_network.keys(): 
        for fid in social_network[uid]: 
            S[uid-1, fid-1] = 1 # 得到的矩阵非常稀疏 options: 迭代删除social用户等 
    uu_social_adj_mat = S
    uu_social_adj_mat = normalize_dense(uu_social_adj_mat)
    
    spR_tr = sp.dok_matrix(R_tr)
    spR_tr = spR_tr.tolil()
    adj_mat_tr = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32) 
    adj_mat_tr = adj_mat_tr.tolil() # convert it to list of lists format 
    adj_mat_tr[:n_users, n_users:] = spR_tr
    adj_mat_tr[n_users:, :n_users] = spR_tr.T
    adj_mat_tr = adj_mat_tr.todok()
    adj_mat_tr = normalize_sp(adj_mat_tr)
    
    spR_val = sp.dok_matrix(R_val)
    spR_val = spR_val.tolil()
    adj_mat_val = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32) 
    adj_mat_val = adj_mat_val.tolil() # convert it to list of lists format 
    adj_mat_val[:n_users, n_users:] = spR_val
    adj_mat_val[n_users:, :n_users] = spR_val.T
    adj_mat_val = adj_mat_val.todok()
    adj_mat_val = normalize_sp(adj_mat_val)
    
    return uu_collab_adj_mat_tr, uu_collab_adj_mat_val, uu_social_adj_mat, adj_mat_tr.tocsr(), adj_mat_val.tocsr()

def datasetsplit(user_ratings, split):
    train_ratings = {}
    test_ratings = {}
    for user in user_ratings:
        size = len(user_ratings[user])
        train_ratings[user] = user_ratings[user][:int(split*size)]   
        test_ratings[user] = user_ratings[user][int(split*size):size]
    return train_ratings, test_ratings

def leave_one_out_split(user_ratings):
    train_ratings = {}
    valid_ratings = {}
    test_ratings = {}
    for user in user_ratings:
        random.shuffle(user_ratings[user])
        size = len(user_ratings[user])
        train_ratings[user] = user_ratings[user][:size-2]
        valid_ratings[user] = user_ratings[user][size-2:size-1] # 返回单元素的list
        test_ratings[user] = user_ratings[user][size-1:size]
        
    return train_ratings, valid_ratings, test_ratings
        
"""
dataloader正确做法: 1.if model 输入 [u,i] pair 保证所有的[u,i]pair被遍历 __len__=数据集所有正样本边个数 2. if model 输入 u and 所有i 保证u被遍历 __len__=数据集所有user个数
总结：总归是训练集所有的正样本边都被训练过
本次我们采用第一种
"""
class myTrainset(Dataset):
    """
    注意idx 
    """
    def __init__(self, config, train_data, neg):
        self.n_items = config['n_items']
        self.n_users = config['n_users']
        self.n_target_users = config['n_target_users']
        self.all_ratings = config['user_rating'] # dict
        self.neg = neg
        train_data_npy = self.get_numpy(train_data) 
        self.train_data_npy = train_data_npy # numpy
    
    def get_numpy(self, train_data):
        train_data_npy = []
        for uid in train_data:
            for item in train_data[uid]:
                train_data_npy.append([uid, item])
        train_data_npy = np.array(train_data_npy)
        return train_data_npy
    
    def __getitem__(self, index):
        """ 
        返回对应index的训练数据 (u,i,[neg个负样本])
        """
        user, pos_item = self.train_data_npy[index][0], self.train_data_npy[index][1] 
        neg_item = np.empty(self.neg, dtype=np.int32)
        for idx in range(self.neg):   
            t = np.random.randint(self.n_users+1, self.n_items+self.n_users+1) # [low, high) itemid: num_of_all_users+1--num_of_nodes
            while t in self.all_ratings[user]: # 不考虑二次负采样
                t = np.random.randint(self.n_users+1, self.n_items+self.n_users+1)
            neg_item[idx] = t-self.n_users-1 # 0开始
        return user-1, pos_item-self.n_users-1, neg_item
    
    def __len__(self): # all u,i pair
        return len(self.train_data_npy)

class myValidset(Dataset):
    
    def __init__(self, config, valid_data, candidate=999):
        self.n_items = config['n_items']
        self.n_users = config['n_users']
        self.n_target_users = config['n_target_users']
        self.all_ratings = config['user_rating'] # dict
        self.n_cnddt = candidate
        self.valid_data = valid_data # dict
    
    def __getitem__(self, user_idx):
        """
        返回对应index的验证数据 (u,i, 999*neg_i)
        """
        [pos_item] = self.valid_data[user_idx+1]
        neg_items = np.empty(self.n_cnddt, dtype=np.int32)
        for idx in range(self.n_cnddt):
            t = np.random.randint(self.n_users+1, self.n_items+self.n_users+1) # [low, high) itemid: num_of_all_users+1--num_of_nodes
            while t in self.all_ratings[user_idx+1]: 
                t = np.random.randint(self.n_users+1, self.n_items+self.n_users+1)
            neg_items[idx] = t-self.n_users-1 # 0开始
        return user_idx, pos_item-self.n_users-1, neg_items
        
    def __len__(self): # all target users
        return len(self.valid_data)

def get_train_loader(config, train_data, args):
    dataset = myTrainset(config, train_data, args.neg)
    # 每次都是随机打乱，然后分成大小为n的若干个mini-batch
    # droplast默认False https://www.cnblogs.com/vvzhang/p/15636814.html
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) # 训练必shuffle防止训练集顺序影响模型训练 测试验证不用 
    return dataloader

def get_valid_loader(config, valid_data, args):
    dataset = myValidset(config, valid_data, 999)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataloader

# 验证dataloader是否正确
# for batch_idx, (user, pos_item, neg_items) in enumerate(tqdm(train_loader, file=sys.stdout)):
    # print(user.size()) # [B]
    # print(pos_item.size()) # [B]
    # print(neg_items.size()) # [B,neg]
#     if int(pos_item)+config['n_users']+1 not in train_data[int(user)+1]:
#         print("wrong")
#     for i in neg_items[0]:       
#         if int(i)+config['n_users']+1 in train_data[int(user)+1]:
#             print("wrong")