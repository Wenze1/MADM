import argparse

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset', default='ciao')
    # parser.add_argument('--data_dir', default='/home/mawenze/raw dataset/ciao/ciao.pkl')
    parser.add_argument('--data_dir', default='../raw dataset/ciao/ciao20230314.pkl')
    # parser.add_argument('--dataset', default='/home/mawenze/raw dataset/flickr')
    # parser.add_argument('--data_dir', default='/home/mawenze/raw dataset/flickr/flickr.pkl')
    # parser.add_argument('--dataset', default='/home/mawenze/raw dataset/yelp')
    # parser.add_argument('--data_dir', default='/home/mawenze/raw dataset/yelp/yelp.pkl')
    parser.add_argument('--base_model_name', default='DESIGN', type=str)
    parser.add_argument('--plugin_name', default="MADM(cl)", type=str)   # MADM(recon)
    
    parser.add_argument('--gpu_id', default=1, type=int)
    # training hyper_parameter
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=200, type=int)
    parser.add_argument('--hop', default=2, type=int) 
    parser.add_argument('--hidden', default=64, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--decay', default=1e-4, type=float)
    
    # GraphLearner hyper_parameter 
    parser.add_argument('--graph_learn_top_k_S', default=30, type=int) 
    parser.add_argument('--graph_learn_epsilon', default=0, type=float)
    parser.add_argument('--graph_skip_conn', default=0.8, type=float)
    parser.add_argument('--graph_learn_num_pers', default=4, type=int)
    parser.add_argument('--metric_type', default='weighted_cosine', type=str)
    parser.add_argument('--graph_learner_hop', default=2, type=int)

    # cl hyper_parameter
    parser.add_argument('--cl_temp', default=0.2, type=float)
    parser.add_argument('--cl_reg', default=1e-6, type=float)
    
    # recon hyper_parameter
    parser.add_argument('--recon_reg', default=0.2, type=float)
    parser.add_argument('--recon_drop', default=0.8, type=float)
    
    # KD hyper_parameter for DESIGN
    parser.add_argument('--kd_reg', default=1, type=float)
    
    
    return parser.parse_args()