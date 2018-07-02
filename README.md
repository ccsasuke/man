# Multinomial Adversarial Nets
This repo contains the source code for our paper:

[**Multinomial Adversarial Networks for Multi-Domain Text Classification**](http://aclweb.org/anthology/N18-1111)
<br>
[Xilun Chen](http://www.cs.cornell.edu/~xlchen/),
[Claire Cardie](http://www.cs.cornell.edu/home/cardie/)
<br>
NAACL-HLT 2018
<br>
[paper](http://aclweb.org/anthology/N18-1111),
[bibtex](https://aclanthology.coli.uni-saarland.de/papers/N18-1111/n18-1111.bib)

## Requirements:
- Python 3.6
- PyTorch 0.4
- PyTorchNet
- scipy
- tqdm (for progress bar)

An Anaconda `environment.yml` file is also provided.

## Dev version

The `dev` branch contains a full version of our code, including some options we experimented but did not include in the final version.

## Before Running

The pre-trained word embeddings file exceeds the 100MB limit of github, and is thus provided as a gzipped tar ball.
Please run the following command to extract it first:

```
tar -xvf data/w2v/word2vec.tar.gz -C data/w2v/
```

## Experiment 1: MDTC on the multi-domain Amazon dataset

```bash
cd code/
python train_man_exp1.py --dataset prep-amazon --model mlp
```

## Experiment 2: Multi-Source Domain Adaptation
```bash
cd code/
# target domain: books
python train_man_exp2.py --dataset prep-amazon --model mlp --no_wgan_trick --domains dvd electronics kitchen --unlabeled_domains books --dev_domains books
# target domain: dvd
python train_man_exp2.py --dataset prep-amazon --model mlp --no_wgan_trick --domains books electronics kitchen --unlabeled_domains dvd --dev_domains dvd
# target domain: electronics
python train_man_exp2.py --dataset prep-amazon --model mlp --no_wgan_trick --domains books dvd kitchen --unlabeled_domains electronics --dev_domains electronics
# target domain: kitchen
python train_man_exp2.py --dataset prep-amazon --model mlp --no_wgan_trick --domains dvd electronics kitchen --unlabeled_domains kitchen --dev_domains kitchen
```

## Experiment 3: MDTC on the FDU-MTL dataset

```bash
cd code/
python train_man_exp3.py --dataset fdu-mtl --model cnn --max_epoch 50
```
A larger batch size can also be used to reduce the training time.
