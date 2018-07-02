import argparse

import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=30)
parser.add_argument('--dataset', default='prep-amazon') # prep-amazon or fdu-mtl
parser.add_argument('--prep_amazon_file', default='../data/prep-amazon/amazon.pkl')
parser.add_argument('--fdu_mtl_dir', default='../data/fdu-mtl/')
# pre-shuffle for cross validation data split
parser.add_argument('--use_preshuffle/', dest='use_preshuffle', action='store_true', default=True)
parser.add_argument('--amazon_preshuffle_file', default='../data/prep-amazon/amazon-shuffle-indices.pkl')
# for preprocessed amazon dataset; set to -1 to use 30000
parser.add_argument('--feature_num', type=int, default=5000)
# labeled domains: if not set, will use default domains for the dataset
parser.add_argument('--domains', type=str, nargs='+', default=[])
parser.add_argument('--unlabeled_domains', type=str, nargs='+', default=[])
parser.add_argument('--dev_domains', type=str, nargs='+', default=[])
parser.add_argument('--emb_filename', default='../data/w2v/word2vec.txt')
parser.add_argument('--kfold', type=int, default=5) # cross-validation (n>=3)
parser.add_argument('--max_seq_len', type=int, default=0) # set to <=0 to not truncate
# which data to be used as unlabeled data: train, unlabeled, or both
parser.add_argument('--unlabeled_data', type=str, default='both')
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--model_save_file', default='./save/man')
parser.add_argument('--test_only', dest='test_only', action='store_true')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--D_learning_rate', type=float, default=0.0001)
parser.add_argument('--fix_emb', action='store_true', default=False)
parser.add_argument('--random_emb', action='store_true', default=False)
# Feature Extractor model: dan or lstm or cnn (for FDU-MTL dataset) or mlp (for prep-amazon dataset)
parser.add_argument('--model', default='mlp')
# for LSTM model
parser.add_argument('--attn', default='dot')  # attention mechanism (for LSTM): avg, last, dot
parser.add_argument('--bdrnn/', dest='bdrnn', action='store_true', default=True)  # bi-directional LSTM
parser.add_argument('--no_bdrnn/', dest='bdrnn', action='store_false', default=True)  # bi-directional LSTM
# for DAN model
parser.add_argument('--sum_pooling/', dest='sum_pooling', action='store_true')
parser.add_argument('--avg_pooling/', dest='sum_pooling', action='store_false')
# for CNN model
parser.add_argument('--kernel_num', type=int, default=200)
parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[3,4,5])
# for MLP model
# F size: feature_num -> F_hidden_sizes -> shared/domain_hidden_size
parser.add_argument('--F_hidden_sizes', type=int, nargs='+', default=[1000, 500])

# gr (gradient reversing, NLL loss in the paper), bs (boundary seeking), l2
# in the paper, we did not talk about the BS loss;
# it's nearly equivalent to the GR (NLL) loss
parser.add_argument('--loss', default='gr')
parser.add_argument('--shared_hidden_size', type=int, default=128)
parser.add_argument('--domain_hidden_size', type=int, default=64)
parser.add_argument('--activation', default='relu') # relu, leaky
parser.add_argument('--F_layers', type=int, default=1)
parser.add_argument('--C_layers', type=int, default=1)
parser.add_argument('--D_layers', type=int, default=1)
parser.add_argument('--wgan_trick/', dest='wgan_trick', action='store_true', default=True)
parser.add_argument('--no_wgan_trick/', dest='wgan_trick', action='store_false')
parser.add_argument('--n_critic', type=int, default=5) # hyperparameter k in the paper
parser.add_argument('--lambd', type=float, default=0.05)
# batch normalization
parser.add_argument('--F_bn/', dest='F_bn', action='store_true', default=False)
parser.add_argument('--no_F_bn/', dest='F_bn', action='store_false')
parser.add_argument('--C_bn/', dest='C_bn', action='store_true', default=True)
parser.add_argument('--no_C_bn/', dest='C_bn', action='store_false')
parser.add_argument('--D_bn/', dest='D_bn', action='store_true', default=True)
parser.add_argument('--no_D_bn/', dest='D_bn', action='store_false')
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--device/', dest='device', type=str, default='cuda')
parser.add_argument('--debug/', dest='debug', action='store_true')
opt = parser.parse_args()

# automatically prepared options
if not torch.cuda.is_available():
    opt.device = 'cpu'

if len(opt.domains) == 0:
    # use default domains
    if opt.dataset.lower() == 'prep-amazon':
        opt.domains = ['books', 'dvd', 'electronics', 'kitchen']
    elif opt.dataset.lower() == 'fdu-mtl':
        opt.domains = ['MR', 'apparel', 'baby', 'books', 'camera_photo', 'dvd', 'electronics', 'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines', 'music', 'software', 'sports_outdoors', 'toys_games', 'video']
    else:
        raise Exception(f'Unknown dataset {opt.dataset}')
opt.all_domains = opt.domains + opt.unlabeled_domains
if len(opt.dev_domains) == 0:
    opt.dev_domains = opt.all_domains

opt.max_kernel_size = max(opt.kernel_sizes)

if opt.activation.lower() == 'relu':
    opt.act_unit = nn.ReLU()
elif opt.activation.lower() == 'leaky':
    opt.act_unit = nn.LeakyReLU()
else:
    raise Exception(f'Unknown activation function {opt.activation}')
