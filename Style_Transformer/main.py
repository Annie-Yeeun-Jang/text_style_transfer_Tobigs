import os
import subprocess

import torch
import time

from data import load_dataset
from models import StyleTransformer, Discriminator
from train import train, auto_eval



import nltk
nltk.download('punkt')

class Config():
    data_path = './data/dataset/'
    log_dir = './log'
    save_path = './save'
    pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Multi' # 'Multi' or 'Cond'
    load_pretrained_embed = True
    min_freq = 3
    max_length = 32
    embed_size = 300
    d_model = 300
    h = 6 # num head
    num_styles = 2
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    num_layers = 8
    batch_size = 16
    num_epoch = 3
    pretrain_lr = 0.0005
    lr_F = 0.0005
    lr_D = 0.0005
    L2 = 0
    iter_D = 5
    iter_F = 10
    F_pretrain_iter = 3000
    log_steps = 5
    eval_steps = 300
    learned_pos_embed = True
    dropout = 0.15
    drop_rate_config = [(1, 0)]
    temperature_config = [(3, 0), (2.5, 700), (2, 1400)]

    slf_factor = 0.25
    cyc_factor = 0.25
    adv_factor = 1

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0

    mask_rate = 0.998
    cyc_rec_loss_iter = 500


def main():
    config = Config()
    train_iters, dev_iters, test_iters, vocab = load_dataset(config) 
    print(config.device)
    print('Vocab size:', len(vocab))
    model_F = StyleTransformer(config, vocab).to(config.device)
    model_D = Discriminator(config, vocab).to(config.device)
    print(config.discriminator_method)

    train(config, vocab, model_F, model_D, train_iters, dev_iters, config.num_epoch)


if __name__ == '__main__':
    main()

