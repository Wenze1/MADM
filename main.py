import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import numpy as np
import os
from utils import get_local_time, mkdir_ifnotexist, early_stopping, get_parameter_number, set_seed, random_neq
import pickle
from arg_parser import parse_args
from dataloader import datasetsplit, get_adj_mat, get_train_loader, get_valid_loader, leave_one_out_split, get_RRT, get_spRRT
from finalnet import MyModel
import torch.utils.tensorboard as tb
from itertools import product



def train(net, optimizer, trainloader, epoch, device):
    net.train()
    # print(net.RRT)
    # print(net.RRT.todense)
    newRRT = net.edge_dropout(net.RRT)
    newRRT = net.sparse_mx_to_torch_sparse_tensor(newRRT)
    net.RRTdrop = newRRT.to_dense()
    # print(net.RRTdrop)
    # print(torch.isnan(net.RRTdrop.to_dense()).any())
    # assert 0
    
    train_loss = 0
    # 此循环一次运行一个batch的数据  features: torch.Size([B,...]) tqdm是运行batch_idx进度 1，2，3，第几个batch进行过计算了
    for batch_idx, (user, pos, neg) in enumerate(tqdm(trainloader, file=sys.stdout)):
        user = user.to(device)  # [B]
        pos_item = pos.to(device)  # [B]
        neg = neg.squeeze(1)
        neg_item = neg.to(device)  # [B, neg]
        l = net.calculate_loss(user, pos_item, neg_item, epoch)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_loss += l.item()
    # writer.add_scalars("BP_4_64_alpha0/Training Loss", {"Social Domain Loss": social_loss, "Item Domain Loss": item_loss}, epoch+1)
    # writer.add_scalar("Training Loss", train_loss, epoch+1)
    print(f'Training on Epoch {epoch + 1}  [train_loss {float(train_loss):f}]')
    return train_loss

# validate 利用评价指标criterion验证模型效果 (也可以用loss) 应该利用全部的验证集 123
def validate(net, config, valid_loader, epoch, device):
    net.eval()
    NDCG_5 = 0.0
    NDCG_10 = 0.0
    NDCG_15 = 0.0
    HT_10 = 0.0
    HT_5 = 0.0
    HT_15 = 0.0
    valid_user = config['n_target_users'] # the number of users in validation
    with torch.no_grad():
        for _ , (user, pos, negs) in enumerate(tqdm(valid_loader, file=sys.stdout)):
            user = user.to(device)  # [B]
            pos_item = pos.to(device)  # [B]
            neg_items = negs.to(device)  # [B, 999]
            # indices = net.batch_full_sort_predict(user, pos_item, neg_items) # [B, 1000] [B,1]
            # indices = indices.cpu().numpy() 
            # # 计算一个batch的rank情况
            HT_5_B, HT_10_B, HT_15_B, NDCG_5_B, NDCG_10_B, NDCG_15_B = net.batch_full_sort_predict(user, pos_item, neg_items, epoch) # compute_rank(indices)
            HT_5 += HT_5_B.item()
            HT_10 += HT_10_B.item()
            HT_15 += HT_15_B.item()
            NDCG_5 += NDCG_5_B.item()
            NDCG_10 += NDCG_10_B.item()
            NDCG_15 += NDCG_15_B.item()
        print(
            f'Validating on epoch {epoch + 1} [HR@5:{float(HT_5 / valid_user):4f} HR@10:{float(HT_10 / valid_user):4f} HR@15:{float(HT_15 / valid_user):4f} NDCG@5:{float(NDCG_5 / valid_user):4f} NDCG@10:{float(NDCG_10 / valid_user):4f} NDCG@15:{float(NDCG_15 / valid_user):4f}]')
        print('--------')
        return HT_5 / valid_user, HT_10 / valid_user, HT_15 / valid_user, NDCG_5 / valid_user, NDCG_10 / valid_user, NDCG_15 / valid_user

if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)
    set_seed(42) # 先声明device再声明seed
    
    # dataset
    data_file = open(args.data_dir, 'rb')
    # 数据集必须处理成这种形式 这样划分训练测试集时才能保证每一个user都有一部分进训练一部分进测试
    # uid: 1--num_of_target_users, num_of_target_users+1--num_of_all_users
    # itemid: num_of_all_users+1--num_of_nodes
    history_u_lists, _, social_adj_lists, _, user_collab, avg_interaction, avg_friend, \
                                                        num_of_target_users, num_of_all_users, num_of_nodes = pickle.load(data_file)
    print("----------------------------")
    print(args.dataset, "statistical information")
    print("num_of_users in u-i and u-u:", num_of_target_users)
    print("num_of_users only exists in u-u:", num_of_all_users-num_of_target_users)
    print("num_of_items:", num_of_nodes-num_of_all_users)
    print("num_of_nodes in the network:", num_of_nodes)
    print("avg_num_of_interaction:", avg_interaction)
    print("avg_num_of_friend:", avg_friend)
    print("----------------------------")  
    
    config = dict()
    config['user_rating'] = history_u_lists
    config['n_users'] = num_of_all_users
    config['n_target_users'] = num_of_target_users
    config['n_items'] = num_of_nodes-num_of_all_users
    config['user_social'] = social_adj_lists
    
    # dataset split
    # train_data, test_data = datasetsplit(history_u_lists, args.split)
    train_data, valid_data, test_data = leave_one_out_split(history_u_lists) # 需要固定随机数种子 否则adj_mat每次都会不一样
    
    # dataloader
    train_loader = get_train_loader(config=config, train_data=train_data, args=args)
    valid_loader = get_valid_loader(config=config, valid_data=valid_data, args=args)
    # load adj mat
    uu_collab_adj_mat_tr, uu_collab_adj_mat_val, uu_social_adj_mat, A_tr, A_val = get_adj_mat(config, args, valid_data, test_data)
    RRT_tr, RRT_val = get_RRT(config, args, valid_data, test_data)
    spRRT_tr, spRRT_val = get_spRRT(config, args, valid_data, test_data)
    # 稀疏矩阵卷积额外写
    config['RRT_tr'] = spRRT_tr # sp
    config['S'] = uu_social_adj_mat # np
    config['A_tr'] = A_tr # sp
    

    # 网格搜索--DESIGN+MADM(cl)
    args.base_model_name = 'DESIGN'
    args.plugin_name = 'MADM(cl)'
    graph_skip_conn = [0.8, 0.9, 0.99, 1]
    cl_reg = [0, 0.001, 0.01, 0.1]
    decay = [1e-7, 1e-4, 1e-3]
    kd_reg = [0, 1]
    
    # # 网格搜索--Diffnet+MADM(recon)
    # args.base_model_name = 'DiffNet++' 
    # args.plugin_name = 'MADM(recon)'
    # graph_skip_conn = [0.8, 0.9, 0.99, 1]
    # recon_reg = [0.1, 0.01, 0.001, 0]     
    # recon_drop = [0.1, 0]
    # decay = [1e-7, 1e-4, 1e-3]
 
    paraspace = product(graph_skip_conn, cl_reg, decay, kd_reg) # 还可以加入list3 list4
    # for j in decay:
    for i, j, k, m in paraspace: # i代表kl_reg  j代表decay
        args.graph_skip_conn = i
        args.cl_reg = j
        args.decay = k
        args.kd_reg = m
        # early stopping parameter
        test_all_step = 1  # test_all_step=x:每x个epoch在验证集上evaluate一次
        best_valid_score = -100000 # best_valid_score和bigger搭配使用 评价指标越大越好
        bigger = True
        conti_step = 0  # 有连续几次验证效果没有超过best
        stopping_step = 10  # 如果连续有n次的验证效果没有超过best 则将early stop
        
        # tensorboard
        t = get_local_time()
        # time_(t)可以回头找log 找其他指标
        dir = f'runs/{args.base_model_name}_{args.plugin_name}_graph_skip_conn{args.graph_skip_conn}_cl_reg{args.cl_reg}_decay{args.decay}_kd_reg{args.kd_reg}_time{t}'
        writer = tb.SummaryWriter(log_dir=dir)
        # 模型
        net = MyModel(config=config, args=args, device=device)
        net = net.to(device)    
        # Learning Algorithm
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate)

        # 模型超参数验证结果保存 网络训练参数保存
        output_dir =  f"./log/{args.dataset}/{t}" # 模型超参数以及结果保存
        mkdir_ifnotexist(output_dir)
        mkdir_ifnotexist('./saved')
        f = open(os.path.join(output_dir, 'logs.txt'), 'w')
        print(get_parameter_number(net))
        f.write(f'模型总参数：{get_parameter_number(net)} \n')
        f.write('-------------------\n')
        f.write('超参数如下\n')
        f.write('-------------------\n')
        f.write('\n'.join([str(k) + ': ' + str(v) for k, v in vars(args).items()]))
        f.write('\n-----------------')
        f.write('\n输出记录如下')
        f.write('\n-----------------')
        
        # 训练 验证
        for epoch in range(args.num_epoch):
            train_loss = train(net, optimizer, train_loader, epoch, device)
            writer.add_scalar("Training Loss", train_loss, epoch+1)
            writer.add_histogram("Weight of User Embeddings", net.user_embs.weight, epoch+1)
            writer.add_histogram("Weight of Structure Learning", net.graphlearner.weight_tensor, epoch+1)
            HT5, HT10, HT15, NDCG5, NDCG10, NDCG15 = validate(
                net, config, valid_loader, epoch, device)
            writer.add_scalar("Testing acc:", HT10, epoch+1)
            a = f'Epoch {epoch+1} HT@5:{HT5:.4f} HT@10:{HT10:.4f} NDCG@5:{NDCG5:.4f} NDCG@10:{NDCG10:.4f}'
            f.write('\n'+a)

            # 早停&模型训练参数保存
            pth_dir = f'../saved/{args.base_model_name}_{args.plugin_name}-{t}.pth'
            valid_result = HT5+HT10
            best_valid_score, conti_step, stop_flag, update_flag = early_stopping(valid_result, best_valid_score,
                                                                                conti_step, stopping_step, bigger)
            if update_flag:
                # torch.save(net.state_dict(), pth_dir)
                print(f'Current best epoch is {epoch + 1}, Model saved in: {pth_dir}')
                print('-------')
            if stop_flag:
                stop_output = 'Finished training, best eval result in epoch %d' % \
                            (epoch + 1 - conti_step * test_all_step)
                print(stop_output)
                f.write('\n'+stop_output)
                break
        f.close()
        writer.close()