# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np

import torch
from torch import cuda
from torch.nn import CrossEntropyLoss

from model import BartModel
from model import BartForMaskedLM
from transformers import BartTokenizer
from transformers.modeling_bart import make_padding_mask

from utils.optim import ScheduledOptim
from utils.helper import optimize, evaluate
from utils.helper import cal_sc_loss, cal_bl_loss
from utils.dataset import read_data, BARTIterator

# TextCNN 모델 불러오기 추가
from classifier.textcnn import EmbeddingLayer, TextCNN
# load_config 불러오기 추가
import yaml
from utils.helper import load_config


def main():

    device = 'cuda' if cuda.is_available() else 'cpu'
    print('[Info] device is {}'.format(device))

    # Load config dictionary
    config_path = "./base.yaml"
    cfg = load_config(config_path)
    opt = cfg['train']

    print('[Info]', opt)

    torch.manual_seed(opt['seed'])


    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    # for token in ['<E>', '<F>']:
    #     tokenizer.add_tokens(token)

    base = BartModel.from_pretrained("facebook/bart-base")
    model = BartForMaskedLM.from_pretrained('facebook/bart-base', config=base.config)
    # model.resize_token_embeddings(len(tokenizer))
    # model.base_model.encoder.embed_tokens=model.base_model.shared
    # model.base_model.decoder.embed_tokens=model.base_model.shared
    # model.lm_head=_make_linear_from_emb(model.base_model.shared)
    # ckpt 가져오기
    # model.load_state_dict(torch.load(cfg['ckpt_path']+cfg['st_ckpt']))
    model.to(device).train()
    print('[Info] Load BartModel')

    # load pre-trained classifier
    cls_name = input("Choose classifier model(BART|TextCNN): ")
    cls = torch.load(cfg['model_path']+cls_name+'classifier.pt')
    cls.to(device).eval()
    
    print('[Info] Load pre-trained classifier')


    train_src, train_tgt, = read_data(opt['style'], opt['max_len'],
                            'train', tokenizer, ratio=float(opt['ratio']))
    valid_src, valid_tgt = read_data(opt['style'], opt['max_len'],
                            'valid', tokenizer)
    print('[Info] {} instances from train set'.format(len(train_src)))
    print('[Info] {} instances from valid set'.format(len(valid_tgt)))

    train_loader, valid_loader = BARTIterator(train_src, train_tgt,
                                                valid_src, valid_tgt, opt)

    loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                            betas=(0.9, 0.98), eps=1e-09), float(opt['lr']), 10000)

    tab = 0
    eval_loss = 1e8
    total_loss_ce = []
    total_loss_sc = []
    total_loss_co = []
    start = time.time()
    train_iter = iter(train_loader)
    print('[Info] Start Training')

    for step in range(opt['save_epoch'],opt['save_epoch']+opt['steps']):

        try:
            batch = next(train_iter) # tuple: (2, batch_size, batch마다 max_len)
        except:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        src, tgt = map(lambda x: x.to(device), batch) # (batch_size, batch마다 max_len)
        src_mask = make_padding_mask(src, tokenizer.pad_token_id) # (batch_size, src의 max_len), pad_token_id = 1, pad token이면 True, 아니면 False로 masking
        src_mask = 1 - src_mask.long() if src_mask is not None else None # masking이 True이면 0, 아니면 1
        logits = model(src, attention_mask=src_mask, decoder_input_ids=tgt)[0] # (batch_size, tgt의 max_len, vocab_size)
        shift_logits = logits[..., :-1, :].contiguous() # y_hat, (batch_size, tgt의 max_len-1, vocab_size), 문장 내 맨 뒤 제거
        shift_labels = tgt[..., 1:].contiguous() # y, (batch_size, tgt의 max_len-1) # 문장 내 맨 앞 제거
        loss_ce = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)) # cross entropy loss
        total_loss_ce.append(loss_ce.item())

        loss_sc, loss_co = torch.tensor(0), torch.tensor(0)
        if opt['sc'] and (200 < step or len(train_loader) < step): # 200 넘으면 계산 시작
            idx = tgt.ne(tokenizer.pad_token_id).sum(-1) # pad token idx 
            loss_sc = cal_sc_loss(logits, idx, cls, tokenizer, opt['style'])
            total_loss_sc.append(float(loss_sc.item())) # style classifier loss
        if opt['bl'] and (200 < step or len(train_loader)< step):
            idx = tgt.ne(tokenizer.pad_token_id).sum(-1)
            loss_co = cal_bl_loss(logits, tgt, idx, tokenizer)
            total_loss_co.append(loss_co.item()) # BLEU-based loss

        optimize(optimizer, loss_ce + loss_sc + loss_co) # loss가 작아지는 방향으로 update

        if step % opt['log_step'] == 0:
            lr = optimizer._optimizer.param_groups[0]['lr']
            print('[Info] steps {:05d} | loss_ce {:.4f} | loss_sc {:.4f} | '
                    'loss_co {:.4f} | lr {:.6f} | second {:.2f}'.format(
                step, np.mean(total_loss_ce), np.mean(total_loss_sc),
                np.mean(total_loss_co), lr, time.time() - start))
            total_loss_ce = []
            total_loss_sc = []
            total_loss_co = []
            start = time.time()

        # if step%1000==0:
        #     torch.save(model.state_dict(), cfg['ckpt_path']+'{}_{}.chkpt'.format(
        #         'bart', step))
        #     print('[Info] The checkpoint file has been updated.')

        if ((len(train_loader) > opt['eval_step']
                and step % opt['eval_step'] == 0)
                or (len(train_loader) < opt['eval_step']
                    and step % len(train_loader) == 0)):
            valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn,
                                                cls, tokenizer, step, opt['style'])
            if eval_loss >= valid_loss: # eval loss가 더 크면
                # torch.save(model.state_dict(), cfg['ckpt_path']+'{}_{}_{}.chkpt'.format(
                #     opt['model'], opt['order'], opt['style']))
                # print('[Info] The checkpoint file has been updated.')
                eval_loss = valid_loss # val loss로 update
                tab = 0
            else: # val이 더 높으면 +1
                tab += 1
            # if tab == opt['patience']: # patience만큼 tab이 나오면 early stop
            #     exit()

    torch.save(model.state_dict(), cfg['ckpt_path']+'bart_modify'+str(opt['save_epoch']+opt['steps'])+'.chkpt')
    # torch.save(model.state_dict(), cfg['ckpt_path']+'bart_modify'+str(opt['save_epoch']+opt['steps'])+'.pt')
    print('[Info] Save BART model.')


if __name__ == '__main__':
    main()
