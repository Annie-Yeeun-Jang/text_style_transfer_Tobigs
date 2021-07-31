import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import argparse

import torch
from torch import cuda
from model import BartModel
from model import BartForMaskedLM
from transformers import BartTokenizer, BartConfig

# load_config 불러오기 추가
import yaml
from utils.helper import load_config


def main():

    device = 'cuda' if cuda.is_available() else 'cpu'
    print('[Info] device is {}'.format(device))

    # Load config dictionary
    config_path = "./base.yaml"
    cfg = load_config(config_path)
    opt = cfg['infer']

    config = BartConfig.from_pretrained('facebook/bart-large')
    config.output_past = True
    config.d_model = 768
    config.encoder_ffn_dim = 3072
    config.decoder_ffn_dim = 3072

    sd = torch.load(cfg['ckpt_path']+cfg['st_ckpt'])
    model = BartForMaskedLM(config)
    model.load_state_dict(sd, strict=False)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    # for token in ['<E>', '<F>']:
    #     tokenizer.add_tokens(token)

    # inference 실행
        # style default 0 --> 1   
    torch.manual_seed(opt['seed'])
    model.config.output_past=True
    model.to(device).eval()

    src_seq = []
    with open(cfg['data_path']+'test.{}'.format(opt['style'])) as fin:
        for line in fin.readlines():
            src_seq.append(line.strip())

    start = time.time()
    with open(cfg['output_path']+'{}.txt'.format(
            'bart_output'), 'w') as fout:
        for idx, line in enumerate(src_seq):
            if idx % 100 == 0:
                print('[Info] processing {} seqs | sencods {:.4f}'.format(
                    idx, time.time() - start))
                start = time.time()
            src = tokenizer.encode(line, return_tensors='pt')

            generated_ids = model.generate(src.to(device),
                                           num_beams=opt['num_beams'],
                                           max_length=opt['length'],
                                           repetition_penalty=opt['repetition_penalty'])

            text = [tokenizer.decode(g)for g in generated_ids][0]
            print(line)
            print(text)
            print()
            fout.write(text.strip() + '\n')


if __name__ == '__main__':
    main()
