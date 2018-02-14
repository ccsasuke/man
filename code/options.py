import argparse

import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=30)
parser.add_argument('--dataset', default='prep-amazon') # prep-amazon or fdu-mtl
parser.add_argument('--prep_amazon_file', default='../data/prep-amazon/amazon.pkl')
parser.add_argument('--fdu_mtl_dir', default='../data/fdu-mtl/')
# pre-shuffle for cross validation data split
parser.add_argument('--use_preshuffle/', dest='use_preshuffle', action='store_true', default=False)
parser.add_argument('--amazon_preshuffle_file', default='../data/prep-amazon/amazon-shuffle-indices.pkl')
# for preprocessed amazon dataset; set to -1 to use 30000
parser.add_argument('--feature_num', type=int, default=5000)
# labeled domains: if not set, will use default domains for the dataset
parser.add_argument('--domains', type=str, nargs='+', default=[])
parser.add_argument('--unlabeled_domains', type=str, nargs='+', default=[])
parser.add_argument('--dev_domains', type=str, nargs='+', default=[])
parser.add_argument('--emb_filename', default='/home/xc253/amdc-data/data/blitzer-amazon/w2v/word2vec.txt')
parser.add_argument('--kfold', type=int, default=5) # cross-validation (n>=3)
parser.add_argument('--max_seq_len', type=int, default=0) # set to <=0 to not truncate
# which data to be used as unlabeled data: train, unlabeled, or both
parser.add_argument('--unlabeled_data', type=str, default='both')
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--model_save_file', default='./save/man')
parser.add_argument('--test_only', dest='test_only', action='store_true')
parser.add_argument('--batch_size', type=int, default=8)
# In PyTorch 0.3, Batch Norm no longer works for size 1 batch,
# so we will skip leftover batch of size < batch_size
parser.add_argument('--no_skip_leftover_batch', dest='skip_leftover_batch', action='store_false', default=True)
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

# gr (gradient reversing), bs (boundary seeking), l2
# in the paper, we did not talk about the BS loss;
# it's nearly equivalent to the GR loss (NLL loss in the paper)
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
# lambda scheduling: not used
parser.add_argument('--lambd_schedule', dest='lambd_schedule', action='store_true', default=False)
# gradient penalty: not used
parser.add_argument('--grad_penalty', dest='grad_penalty', default='none') #none, wgan or dragan
parser.add_argument('--onesided_gp', dest='onesided_gp', action='store_true')
parser.add_argument('--gp_lambd', type=float, default=0.1)
# orthogality penalty: not used
parser.add_argument('--ortho_penalty', dest='ortho_penalty', type=float, default=0)
# normalizing F features: not used
parser.add_argument('--F_normalize/', dest='F_normalize', action='store_true')
parser.add_argument('--F_logsoftmax/', dest='F_logsoftmax', action='store_true')
# batch normalization
parser.add_argument('--F_bn/', dest='F_bn', action='store_true', default=False)
parser.add_argument('--no_F_bn/', dest='F_bn', action='store_false')
parser.add_argument('--C_bn/', dest='C_bn', action='store_true', default=True)
parser.add_argument('--no_C_bn/', dest='C_bn', action='store_false')
parser.add_argument('--D_bn/', dest='D_bn', action='store_true', default=True)
parser.add_argument('--no_D_bn/', dest='D_bn', action='store_false')
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--use_cuda/', dest='use_cuda', action='store_true', default=True)
parser.add_argument('--debug/', dest='debug', action='store_true')
opt = parser.parse_args()

# automatically prepared options
opt.use_cuda = opt.use_cuda and torch.cuda.is_available()

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
