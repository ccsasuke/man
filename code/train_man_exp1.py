from collections import defaultdict
import itertools
import logging
import math
import os
import pickle
import random
import sys
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchnet.meter import ConfusionMeter

from options import opt
from data_prep.msda_preprocessed_amazon_dataset import get_msda_amazon_datasets
from models import *
import utils

if opt.random_seed >= 0:
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)

# save models and logging
if not os.path.exists(opt.model_save_file):
    os.makedirs(opt.model_save_file)
log = logging.getLogger(__name__)

def train(train_sets, dev_sets, test_sets, unlabeled_sets, fold):
    """
    train_sets, dev_sets, test_sets: dict[domain] -> TensorDataset
    For unlabeled domains, no train_sets are available
    """
    # dataset loaders
    train_loaders, unlabeled_loaders = {}, {}
    train_iters, unlabeled_iters = {}, {}
    dev_loaders, test_loaders = {}, {}
    for domain in opt.domains:
        train_loaders[domain] = DataLoader(train_sets[domain],
                opt.batch_size, shuffle=True)
        train_iters[domain] = iter(train_loaders[domain])
    for domain in opt.all_domains:
        dev_loaders[domain] = DataLoader(dev_sets[domain],
                opt.batch_size, shuffle=False)
        test_loaders[domain] = DataLoader(test_sets[domain],
                opt.batch_size, shuffle=False)
        if domain in opt.unlabeled_domains:
            uset = unlabeled_sets[domain]
        else:
            # for labeled domains, consider which data to use as unlabeled set
            if opt.unlabeled_data == 'both':
                uset = ConcatDataset([train_sets[domain], unlabeled_sets[domain]])
            elif opt.unlabeled_data == 'unlabeled':
                uset = unlabeled_sets[domain]
            elif opt.unlabeled_data == 'train':
                uset = train_sets[domain]
            else:
                raise Exception(f'Unknown options for the unlabeled data usage: {opt.unlabeled_data}')
        unlabeled_loaders[domain] = DataLoader(uset,
                opt.batch_size, shuffle=True)
        unlabeled_iters[domain] = iter(unlabeled_loaders[domain])

    # models
    F_s = None
    F_d = {}
    C, D = None, None
    if opt.model.lower() == 'mlp':
        F_s = MlpFeatureExtractor(opt.feature_num, opt.F_hidden_sizes,
                opt.shared_hidden_size, opt.dropout, opt.F_bn)
        for domain in opt.domains:
            F_d[domain] = MlpFeatureExtractor(opt.feature_num, opt.F_hidden_sizes,
                opt.domain_hidden_size, opt.dropout, opt.F_bn)
    else:
        raise Exception(f'Unknown model architecture {opt.model}')
    C = SentimentClassifier(opt.C_layers, opt.shared_hidden_size + opt.domain_hidden_size,
            opt.shared_hidden_size + opt.domain_hidden_size, opt.num_labels,
            opt.dropout, opt.C_bn)
    D = DomainClassifier(opt.D_layers, opt.shared_hidden_size, opt.shared_hidden_size,
            len(opt.all_domains), opt.loss, opt.dropout, opt.D_bn)
    
    F_s, C, D = F_s.to(opt.device), C.to(opt.device), D.to(opt.device)
    for f_d in F_d.values():
        f_d = f_d.to(opt.device)
    # optimizers
    optimizer = optim.Adam(itertools.chain(*map(list, [F_s.parameters() if F_s else [], C.parameters()] + [f.parameters() for f in F_d.values()])), lr=opt.learning_rate)
    optimizerD = optim.Adam(D.parameters(), lr=opt.D_learning_rate)

    # testing
    if opt.test_only:
        log.info(f'Loading model from {opt.model_save_file}...')
        F_s.load_state_dict(torch.load(os.path.join(opt.model_save_file,
                                                    f'netF_s_fold{fold}.pth')))
        for domain in opt.all_domains:
            if domain in F_d:
                F_d[domain].load_state_dict(torch.load(os.path.join(opt.model_save_file,
                        f'net_F_d_{domain}_fold{fold}.pth')))
        C.load_state_dict(torch.load(os.path.join(opt.model_save_file,
                                                  f'netC_fold{fold}.pth')))
        D.load_state_dict(torch.load(os.path.join(opt.model_save_file,
                                                  f'netD_fold{fold}.pth')))

        log.info('Evaluating validation sets:')
        acc = {}
        for domain in opt.all_domains:
            acc[domain] = evaluate(domain, dev_loaders[domain],
                                   F_s, F_d[domain] if domain in F_d else None, C)
        avg_acc = sum([acc[d] for d in opt.dev_domains]) / len(opt.dev_domains)
        log.info(f'Average validation accuracy: {avg_acc}')
        log.info('Evaluating test sets:')
        test_acc = {}
        for domain in opt.all_domains:
            test_acc[domain] = evaluate(domain, test_loaders[domain],
                    F_s, F_d[domain] if domain in F_d else None, C)
        avg_test_acc = sum([test_acc[d] for d in opt.dev_domains]) / len(opt.dev_domains)
        log.info(f'Average test accuracy: {avg_test_acc}')
        return {'valid': acc, 'test': test_acc}

    # training
    best_acc, best_avg_acc = defaultdict(float), 0.0

    for epoch in range(opt.max_epoch):
        F_s.train()
        C.train()
        D.train()
        for f in F_d.values():
            f.train()
            
        # training accuracy
        correct, total = defaultdict(int), defaultdict(int)
        # D accuracy
        d_correct, d_total = 0, 0
        # conceptually view 1 epoch as 1 epoch of the first domain
        num_iter = len(train_loaders[opt.domains[0]])
        for i in tqdm(range(num_iter)):
            # D iterations
            utils.freeze_net(F_s)
            map(utils.freeze_net, F_d.values())
            utils.freeze_net(C)
            utils.unfreeze_net(D)
            # optional WGAN n_critic trick
            n_critic = opt.n_critic
            if opt.wgan_trick:
                if opt.n_critic>0 and ((epoch==0 and i<25) or i%500==0):
                    n_critic = 100

            for _ in range(n_critic):
                D.zero_grad()
                loss_d = {}
                # train on both labeled and unlabeled domains
                for domain in opt.all_domains:
                    # targets not used
                    d_inputs, _ = utils.endless_get_next_batch(
                            unlabeled_loaders, unlabeled_iters, domain)
                    d_targets = utils.get_domain_label(opt.loss, domain, len(d_inputs))
                    shared_feat = F_s(d_inputs)
                    d_outputs = D(shared_feat)
                    # D accuracy
                    _, pred = torch.max(d_outputs, 1)
                    d_total += len(d_inputs)
                    if opt.loss.lower() == 'l2':
                        _, tgt_indices = torch.max(d_targets, 1)
                        d_correct += (pred==tgt_indices).sum().item()
                        l_d = functional.mse_loss(d_outputs, d_targets)
                        l_d.backward()
                    else:
                        d_correct += (pred==d_targets).sum().item()
                        l_d = functional.nll_loss(d_outputs, d_targets)
                        l_d.backward()
                    loss_d[domain] = l_d.item()
                optimizerD.step()

            # F&C iteration
            utils.unfreeze_net(F_s)
            map(utils.unfreeze_net, F_d.values())
            utils.unfreeze_net(C)
            utils.freeze_net(D)
            if opt.fix_emb:
                utils.freeze_net(F_s.word_emb)
                map(utils.freeze_net, F_d.values())
            F_s.zero_grad()
            for f_d in F_d.values():
                f_d.zero_grad()
            C.zero_grad()
            shared_feats, domain_feats = [], []
            for domain in opt.domains:
                inputs, targets = utils.endless_get_next_batch(
                        train_loaders, train_iters, domain)
                targets = targets.to(opt.device)
                shared_feat = F_s(inputs)
                shared_feats.append(shared_feat)
                domain_feat = F_d[domain](inputs)
                domain_feats.append(domain_feat)
                features = torch.cat((shared_feat, domain_feat), dim=1)
                c_outputs = C(features)
                l_c = functional.nll_loss(c_outputs, targets)
                l_c.backward(retain_graph=True)
                # training accuracy
                _, pred = torch.max(c_outputs, 1)
                total[domain] += targets.size(0)
                correct[domain] += (pred == targets).sum().item()
            # update F with D gradients on all domains
            for domain in opt.all_domains:
                d_inputs, _ = utils.endless_get_next_batch(
                        unlabeled_loaders, unlabeled_iters, domain)
                shared_feat = F_s(d_inputs)
                d_outputs = D(shared_feat)
                if opt.loss.lower() == 'gr':
                    d_targets = utils.get_domain_label(opt.loss, domain, len(d_inputs))
                    l_d = functional.nll_loss(d_outputs, d_targets)
                    log.debug(f'D loss: {l_d.item()}')
                    if opt.lambd > 0:
                        l_d *= -opt.lambd
                elif opt.loss.lower() == 'bs':
                    d_targets = utils.get_random_domain_label(opt.loss, len(d_inputs))
                    l_d = functional.kl_div(d_outputs, d_targets, size_average=False)
                    if opt.lambd > 0:
                        l_d *= opt.lambd
                elif opt.loss.lower() == 'l2':
                    d_targets = utils.get_random_domain_label(opt.loss, len(d_inputs))
                    l_d = functional.mse_loss(d_outputs, d_targets)
                    if opt.lambd > 0:
                        l_d *= opt.lambd
                l_d.backward()
                if opt.model.lower() != 'lstm':
                    log.debug(f'F_s norm: {F_s.net[-2].weight.grad.data.norm()}')

            optimizer.step()

        # end of epoch
        log.info('Ending epoch {}'.format(epoch+1))
        if d_total > 0:
            log.info('D Training Accuracy: {}%'.format(100.0*d_correct/d_total))
        log.info('Training accuracy:')
        log.info('\t'.join(opt.domains))
        log.info('\t'.join([str(100.0*correct[d]/total[d]) for d in opt.domains]))
        log.info('Evaluating validation sets:')
        acc = {}
        for domain in opt.all_domains:
            acc[domain] = evaluate(domain, dev_loaders[domain],
                                   F_s, F_d[domain] if domain in F_d else None, C)
        avg_acc = sum([acc[d] for d in opt.dev_domains]) / len(opt.dev_domains)
        log.info(f'Average validation accuracy: {avg_acc}')
        log.info('Evaluating test sets:')
        test_acc = {}
        for domain in opt.all_domains:
            test_acc[domain] = evaluate(domain, test_loaders[domain],
                    F_s, F_d[domain] if domain in F_d else None, C)
        avg_test_acc = sum([test_acc[d] for d in opt.dev_domains]) / len(opt.dev_domains)
        log.info(f'Average test accuracy: {avg_test_acc}')

        if avg_acc > best_avg_acc:
            log.info(f'New best average validation accuracy: {avg_acc}')
            best_acc['valid'] = acc
            best_acc['test'] = test_acc
            best_avg_acc = avg_acc
            with open(os.path.join(opt.model_save_file, 'options.pkl'), 'wb') as ouf:
                pickle.dump(opt, ouf)
            torch.save(F_s.state_dict(),
                    '{}/netF_s_fold{}.pth'.format(opt.model_save_file, fold))
            for d in opt.domains:
                if d in F_d:
                    torch.save(F_d[d].state_dict(),
                            '{}/net_F_d_{}_fold{}.pth'.format(opt.model_save_file, d, fold))
            torch.save(C.state_dict(),
                    '{}/netC_fold{}.pth'.format(opt.model_save_file, fold))
            torch.save(D.state_dict(),
                    '{}/netD_fold{}.pth'.format(opt.model_save_file, fold))

    # end of training
    log.info(f'Best average validation accuracy: {best_avg_acc}')
    return best_acc


def evaluate(name, loader, F_s, F_d, C):
    F_s.eval()
    if F_d:
        F_d.eval()
    C.eval()
    it = iter(loader)
    correct = 0
    total = 0
    confusion = ConfusionMeter(opt.num_labels)
    for inputs, targets in tqdm(it):
        targets = targets.to(opt.device)
        if not F_d:
            # unlabeled domain
            d_features = torch.zeros(len(targets), opt.domain_hidden_size).to(opt.device)
        else:
            d_features = F_d(inputs)
        features = torch.cat((F_s(inputs), d_features), dim=1)
        outputs = C(features)
        _, pred = torch.max(outputs, 1)
        confusion.add(pred.data, targets.data)
        total += targets.size(0)
        correct += (pred == targets).sum().item()
    acc = correct / total
    log.info('{}: Accuracy on {} samples: {}%'.format(name, total, 100.0*acc))
    log.debug(confusion.conf)
    return acc


def cross_validation(kfold):
    datasets = {}
    unlabeled_sets = {}
    log.info(f'Loading {opt.dataset} Datasets...')
    for domain in opt.domains + opt.unlabeled_domains:
        datasets[domain], unlabeled_sets[domain] = get_msda_amazon_datasets(
                opt.prep_amazon_file, domain, kfold, opt.feature_num)
    opt.num_labels = 2
    log.info(f'Done Loading {opt.dataset} Datasets.')
    log.info(f'Domains: {opt.domains}')

    cv = {}
    log.info(f'Starting {kfold}-fold Cross Validation...')
    for fold in range(kfold):
        log.info(f'Starting fold {fold}...')
        train_sets, dev_sets, test_sets = {}, {}, {}
        for domain in opt.all_domains:
            lset = datasets[domain]
            if domain in opt.domains:
                # only labeled domains have training data
                train_sets[domain] = lset.get_trainset(fold)
            dev_sets[domain] = lset.get_devset(fold)
            test_sets[domain] = lset.get_testset(fold)
        cv[fold] = train(train_sets, dev_sets, test_sets, unlabeled_sets, fold)
        log.info(f'Ending fold {fold}...')
        acc = sum(cv[fold]['valid'].values()) / len(cv[fold]['valid'])
        log.info(f'Validation Set Domain Average\t{acc}')
        test_acc = sum(cv[fold]['test'].values()) / len(cv[fold]['test'])
        log.info(f'Test Set Domain Average\t{test_acc}')

    avg_acc = utils.average_cv_accuracy(cv)
    avg_acc, avg_acc_test = avg_acc['valid'], avg_acc['test']
    log.info(f'{kfold}-Fold Cross Validation Accuracies:')
    for domain in opt.all_domains:
        log.info(f'{domain}\t{avg_acc[domain]}')
    overall = sum(avg_acc.values()) / len(avg_acc)
    log.info(f'Overall Validation Set Average\t{overall}')
    log.info(f'{kfold}-Fold Cross Validation Accuracies on Test Set:')
    for domain in opt.all_domains:
        log.info(f'{domain}\t{avg_acc_test[domain]}')
    overall_test = sum(avg_acc_test.values()) / len(avg_acc_test)
    log.info(f'Overall Test Set Average\t{overall_test}')
    return overall
    

def main():
    if not os.path.exists(opt.model_save_file):
        os.makedirs(opt.model_save_file)

    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
    fh = logging.FileHandler(os.path.join(opt.model_save_file, 'log.txt'))
    log.addHandler(fh)

    # output options
    log.info(opt)

    accuracy = cross_validation(opt.kfold)
    log.removeHandler(fh)
    return accuracy


if __name__ == '__main__':
    main()
