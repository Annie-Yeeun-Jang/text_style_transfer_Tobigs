import os
import torch
import numpy as np
from torch import nn, optim
import time
from torch.nn.utils import clip_grad_norm_

from evaluator import Evaluator
from utils import tensor2text, calc_ppl, idx2onehot, add_noise, word_drop
from train import get_lengths
#from tensorboardX import SummaryWriter
#from torch.nn.utils import clip_grad_norm_

from evaluator import Evaluator
from utils import tensor2text, calc_ppl, idx2onehot, add_noise, word_drop

from data import load_dataset
from models import StyleTransformer, Discriminator
from train import train #, auto_eval #그냥 이거 가져와서 쓸까...? test iter만 다시 주면 되니까
from main import Config

def final_eval(config, vocab, model_F, test_iters, global_step, temperature):
    model_F.eval()
    #vocab_size = len(vocab)
    eos_idx = vocab.stoi['<eos>']

    def inference(data_iter, raw_style, masking_rate):
        gold_text = []
        raw_output = []
        rev_output = []
        for batch in data_iter:
            mask = np.random.random_sample()
            if mask < masking_rate:
                continue

            inp_tokens = batch.text
            inp_lengths = get_lengths(inp_tokens, eos_idx)
            raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
            rev_styles = 1 - raw_styles

            with torch.no_grad():
                raw_log_probs = model_F(
                    inp_tokens,
                    None,
                    inp_lengths,
                    raw_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )

            with torch.no_grad():
                rev_log_probs = model_F(
                    inp_tokens,
                    None,
                    inp_lengths,
                    rev_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )

            gold_text += tensor2text(vocab, inp_tokens.cpu())
            raw_output += tensor2text(vocab, raw_log_probs.argmax(-1).cpu())
            rev_output += tensor2text(vocab, rev_log_probs.argmax(-1).cpu())

        return gold_text, raw_output, rev_output

    native_iter = test_iters.native_iter
    nonnative_iter = test_iters.nonnative_iter

    gold_text, raw_output, rev_output = zip(inference(nonnative_iter, 0, config.mask_rate), inference(native_iter, 1, config.mask_rate))
   
    # save output
    save_file = config.save_folder + '/' + str(global_step) + '.txt'
    
    with open(save_file, 'w') as fw:
        for idx in range(len(rev_output[0])):
            print('*' * 20, 'nonnative sample', '*' * 20, file=fw)
            print('[gold]', gold_text[0][idx], file=fw)
            print('[raw ]', raw_output[0][idx], file=fw)
            print('[rev ]', rev_output[0][idx], file=fw)

        print('*' * 20, '********', '*' * 20, file=fw)

        for idx in range(len(rev_output[1])):
            print('*' * 20, 'native sample', '*' * 20, file=fw)
            print('[gold]', gold_text[1][idx], file=fw)
            print('[raw ]', raw_output[1][idx], file=fw)
            print('[rev ]', rev_output[1][idx], file=fw)

        print('*' * 20, '********', '*' * 20, file=fw)

    model_F.train()


def main():
    config = Config()
    train_iters, dev_iters, test_iters, vocab = load_dataset(config)
    print(config.device)
    print('Vocab size:', len(vocab))
    #model_F에 모델 가중치 불러오기 !!
    model_F = StyleTransformer(config, vocab).to(config.device)
    #model_D = Discriminator(config, vocab).to(config.device)
    saved_path = './save/Jul15220935/ckpts/2100_F.pth'
    checkpoint = torch.load(saved_path)
    model_F.load_state_dict(checkpoint)
    print(config.discriminator_method)

    #train(config, vocab, model_F, model_D, train_iters, dev_iters, config.num_epoch)
    #train 대신 auto_eval
    # global_step은 저장할때 이름 기록용으로만 쓰는거라 아무거나 둬도 괜찮음
    # temperature는 forward에만 쓰임
    
    #config.save_folder = config.save_path + '/' + str(time.strftime('%b%d%H%M%S', time.localtime()))
    config.save_folder = config.save_path + '/final_infer'
    final_eval(config, vocab, model_F, test_iters, global_step = 0, temperature = 1.0)


if __name__ == '__main__':
    main()
