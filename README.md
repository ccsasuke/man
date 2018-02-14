# Multinomial Adversarial Nets
This repo contains the source code for our paper:

[**Multinomial Adversarial Networks for Multi-Domain Text Classification**]()
<br>
[Xilun Chen](http://www.cs.cornell.edu/~xlchen/),
[Claire Cardie](http://www.cs.cornell.edu/home/cardie/)
<br>
NAACL 2018 (To appear)

## Requirements:
- Python 3.6
- PyTorch 0.3
- PyTorchNet
- tqdm (for progress bar)

A Anaconda `environment.yml` file is also provided.

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
python train_man_exp3.py --dataset fdu-mtl --model cnn --fdu-mtl_dir {dataset_path} --max_epoch 50
```
A larger batch size can also be used to reduce the training time.
