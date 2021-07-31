import os
import time
import torch
import numpy as np
from torch import nn, optim

from torch.nn.utils import clip_grad_norm_

from evaluator import Evaluator
from utils import tensor2text, calc_ppl, idx2onehot, add_noise, word_drop


def get_lengths(tokens, eos_idx): # token shape : (batch_size, max_len)
    lengths = torch.cumsum(tokens == eos_idx, 1) # 문장에서 <eos>가 등장하기 전까진 0으로 된 텐서 생성 shape : (batch_size, max_len)
    lengths = (lengths == 0).long().sum(-1) # 각 문장의 0의 갯수 센 벡터 생성, 0의 개수 == 해당 문장의 길이 shape : (batch_size)
    lengths = lengths + 1 # +1 for <eos> token # 마지막 <eos> 까지 길이에 포함
    return lengths # 각 문장의 길이 벡터 return

def batch_preprocess(batch, pad_idx, eos_idx, reverse=False): # batch : (positive data, negative data)// positive data, negative data shape (batch_size_pos, max_len_pos), (batch_size_neg, max_len_neg)
    batch_native, batch_nonnative = batch # batch를 label에 따라 나눔
    diff = batch_native.size(1) - batch_nonnative.size(1) # pos와 neg의 문장 길이가 같은지 확인, diff == 문장 길이 차이
    if diff < 0: # pos가 neg보다 짧을 경우 
        print("negative sentences are longer than positive sentences")
        pad = torch.full_like(batch_nonnative[:, :-diff], pad_idx) # 부족한 pos의 문장 길이를 <pad>로 채움
        batch_native = torch.cat((batch_native, pad), 1) # neg와 pos의 길이 같아짐
    elif diff > 0: # pos가 neg보다 길 경우
        print("positive sentences are longer than negative sentences")
        pad = torch.full_like(batch_native[:, :diff], pad_idx) # 부족한 neg의 문장 길이를 <pad>로 채움
        batch_nonnative = torch.cat((batch_nonnative, pad), 1) # neg와 pos의 길이 같아짐

    native_styles = torch.ones_like(batch_native[:, 0]) # pos label == 1 shape : (batch_size_pos, 1)
    nonnative_styles = torch.zeros_like(batch_nonnative[:, 0]) # neg label == 0 shape : (batch_size_neg, 0)

    if reverse: # batch 안에 순서가 반대일 경우 
        batch_native, batch_nonnative = batch_nonnative, batch_native
        native_styles, nonnative_styles = nonnative_styles, native_styles
        
    tokens = torch.cat((batch_native, batch_nonnative), 0) # 텍스트 데이터 합치기 shape : (batch_size_pos + batch_size_neg, max_len)
    lengths = get_lengths(tokens, eos_idx) # 각 문장의 길이 벡터
    styles = torch.cat((native_styles, nonnative_styles), 0) # label 합치기 : shape : (batch_size_pos + batch_size_neg, max_len)

    return tokens, lengths, styles
        

def d_step(config, vocab, model_F, model_D, optimizer_D, batch, temperature): # discriminator 학습 함수
    model_F.eval() # generator는 inference
    pad_idx = vocab.stoi['<pad>'] # pad 인덱스
    eos_idx = vocab.stoi['<eos>'] # eos 인덱스
    vocab_size = len(vocab)
    loss_fn = nn.NLLLoss(reduction='none') # 손실함수는 negative log liklihood

    inp_tokens, inp_lengths, raw_styles = batch_preprocess(batch, pad_idx, eos_idx)
    rev_styles = 1 - raw_styles # label을 반대로 한 label
    batch_size = inp_tokens.size(0) 

    with torch.no_grad(): # generator는 각 토큰의 softmax를 return
        raw_gen_log_probs = model_F( # x와 s로 생성된 문장
            inp_tokens, 
            None, # inference이기 때문에 gold token 없음
            inp_lengths,
            raw_styles, # 본래 스타일 s
            generate=True,
            differentiable_decode=True, # soft embedding
            temperature=temperature, 
        )
        rev_gen_log_probs = model_F( # x와 s_hat으로 생성된 문장
            inp_tokens,
            None,
            inp_lengths,
            rev_styles, # 반대 스타일 s_hat
            generate=True,
            differentiable_decode=True,
            temperature=temperature,
        )

    
    raw_gen_soft_tokens = raw_gen_log_probs.exp() # 본래 log prob이 나오기 때문에 다시 exp
    raw_gen_lengths = get_lengths(raw_gen_soft_tokens.argmax(-1), eos_idx) # 생성된 문장의 길이 벡터

    
    rev_gen_soft_tokens = rev_gen_log_probs.exp()
    rev_gen_lengths = get_lengths(rev_gen_soft_tokens.argmax(-1), eos_idx)

        

    if config.discriminator_method == 'Multi': # Multi-class discriminator일 경우 input : 문장, output : style의 softmax dist.
        gold_log_probs = model_D(inp_tokens, inp_lengths) # 원래 문장 x를 넣었을 때의 dist.
        gold_labels = raw_styles + 1 # label 0 : 생성문장 1 : neg, 2 : pos

        raw_gen_log_probs = model_D(raw_gen_soft_tokens, raw_gen_lengths) # x와 s로 만든 문장의 dist. shape : (batch_size, class_num)
        rev_gen_log_probs = model_D(rev_gen_soft_tokens, rev_gen_lengths) # x와 s_hat으로 만든 문장의 dist. shape : (batch_size, class_num)
        gen_log_probs = torch.cat((raw_gen_log_probs, rev_gen_log_probs), 0) # raw와 rev를 하나로 합침 shape : (2*batch_size, class_num)
        raw_gen_labels = raw_styles + 1 # label 0 : 생성문장 1 : neg, 2 : pos
        rev_gen_labels = torch.zeros_like(rev_styles) # rev는 생성 문장이므로 모두 label이 0
        gen_labels = torch.cat((raw_gen_labels, rev_gen_labels), 0) # raw와 rev를 하나로 합침 shape : (2*batch_size, 1)
    else: # conditional discriminator일 경우 input : 문장, s, output : s와 문장의 일치 여부 
        raw_gold_log_probs = model_D(inp_tokens, inp_lengths, raw_styles) # x와 s 삽입시 일치 여부 shape : (batch_size, 1)
        rev_gold_log_probs = model_D(inp_tokens, inp_lengths, rev_styles) # x와 s_hat 삽입시 일치 여부 shape : (batch_size, 1)
        gold_log_probs = torch.cat((raw_gold_log_probs, rev_gold_log_probs), 0) # 위의 두개를 합친 것 shape : (2*batch_size, 1)
        raw_gold_labels = torch.ones_like(raw_styles) # 일치 할 때 label : 1
        rev_gold_labels = torch.zeros_like(rev_styles) # 일치 하지 않을 때 label : 0
        gold_labels = torch.cat((raw_gold_labels, rev_gold_labels), 0) # 위의 두개를 합친 것 shape : (2*batch_size, 1)

        
        raw_gen_log_probs = model_D(raw_gen_soft_tokens, raw_gen_lengths, raw_styles) # 생성 문장 x_hat과 그에 상응하는 s의 일치 여부 shape : (batch_size, 1)
        rev_gen_log_probs = model_D(rev_gen_soft_tokens, rev_gen_lengths, rev_styles) # 생성 문장 x_hat과 그에 반대되는 s_rev의 일치 여부 shape : (batch_size, 1)
        gen_log_probs = torch.cat((raw_gen_log_probs, rev_gen_log_probs), 0) # 위의 두개를 합친 것 shape : (2*batch_size, 1)
        raw_gen_labels = torch.ones_like(raw_styles) # 생성 문장 x_hat과 그에 상응하는 s이므로 label : 1 shape : (batch_size, 1)
        rev_gen_labels = torch.zeros_like(rev_styles) # 생성 문장 x_hat과 그에 반대되는 s_rev이므로 label : 0 shape : (batch_size, 1)
        gen_labels = torch.cat((raw_gen_labels, rev_gen_labels), 0) # 위의 두개를 합친 것 shape : (2*batch_size, 1)

    
    adv_log_probs = torch.cat((gold_log_probs, gen_log_probs), 0) # 모든 데이터 합치기 shape : (4*batch_size, 1)
    adv_labels = torch.cat((gold_labels, gen_labels), 0) # shape : (4*batch_size, 1)
    adv_loss = loss_fn(adv_log_probs, adv_labels) # Cross entropy 계산
    assert len(adv_loss.size()) == 1 # 최종 loss가 벡터인지 확인
    adv_loss = adv_loss.sum() / batch_size # 최종 loss 계산
    loss = adv_loss
    
    optimizer_D.zero_grad() 
    loss.backward() # 역전파
    clip_grad_norm_(model_D.parameters(), 5) # gradient clipping 사용, max_norm : 5
    optimizer_D.step() # update

    model_F.train() # 다시 train 모드로

    return adv_loss.item() # loss return

def f_step(config, vocab, model_F, model_D, optimizer_F, batch, temperature, drop_decay, # generator 학습 함수, 3가지 손실 함수로 구성 (self reconstruction, cycle consistency, style consistency)
           cyc_rec_enable=True, cyc_rec = True):
    model_D.eval() # discriminator inference 모드 
    
    pad_idx = vocab.stoi['<pad>'] # <pad> 인덱스 
    eos_idx = vocab.stoi['<eos>'] # <eos> 인덱스
    unk_idx = vocab.stoi['<unk>'] # <unk> 인덱스 
    vocab_size = len(vocab)
    loss_fn = nn.NLLLoss(reduction='none') # loss 는 CE

    inp_tokens, inp_lengths, raw_styles = batch_preprocess(batch, pad_idx, eos_idx) # input token size : (batch_size, max_len), raw_styles shape : (batch_size, 1)
    rev_styles = 1 - raw_styles # 실제 label과 반대 label
    batch_size = inp_tokens.size(0) 
    token_mask = (inp_tokens != pad_idx).float() # 실제 token mask 생성

    optimizer_F.zero_grad()

    # self reconstruction loss

    noise_inp_tokens = word_drop( # 토큰을 inp_drop_prob으로 unk로 만듬. 
        inp_tokens,
        inp_lengths, 
        config.inp_drop_prob * drop_decay,
        vocab
    )
    noise_inp_lengths = get_lengths(noise_inp_tokens, eos_idx)

    slf_log_probs = model_F( # 인코더에 noise_inp_token을 디코더에 inp_token을 넣어 문장 생성 # shape : (batch_size, seq_len, vocab_size)
        noise_inp_tokens, 
        inp_tokens, 
        noise_inp_lengths,
        raw_styles,
        generate=False, # 훈련 모드 
        differentiable_decode=False, # 디코더에서 토큰을 입력으로 함. 
        temperature=temperature,
    )

    slf_rec_loss = loss_fn(slf_log_probs.transpose(1, 2), inp_tokens) * token_mask # * : element-wise 실제 input과 생성된 문장의 차이 비교
    slf_rec_loss = slf_rec_loss.sum() / batch_size
    slf_rec_loss *= config.slf_factor # 로스 계산
    
    slf_rec_loss.backward() # 역전파

    # cycle consistency loss // x - > x_hat -> x로 생성했을 때, x와 생성된 x의 차이

    if not cyc_rec_enable: # cycle consistency loss를 사용할 수 없다면 여기서 멈춤
        optimizer_F.step()
        model_D.train()
        return slf_rec_loss.item(), 0, 0
    
    gen_log_probs = model_F( # x -> x_hat을 만들기
        inp_tokens,
        None, # inference해야 하기 때문에 gold token 없음
        inp_lengths,
        rev_styles, # s_hat으로 만들기
        generate=True,
        differentiable_decode=True, # gold token이 없기 때문에 soft embedding 사용
        temperature=temperature,
    )

    gen_soft_tokens = gen_log_probs.exp() # 문장 내 토큰 별 dist. 생성 shape : (batch_size, seq_len, vocab)
    gen_lengths = get_lengths(gen_soft_tokens.argmax(-1), eos_idx) # (batch_size)
    if cyc_rec == True:

        cyc_log_probs = model_F( # x_hat을 다시 x로 돌리기 (batch_size, seq_len, vocab_size)
            gen_soft_tokens, # x_hat, 여기선 token dist.
            inp_tokens, # 본래의 토큰을 gold token으로 사용
            gen_lengths,
            raw_styles, # s로 style transfer
            generate=False,
            differentiable_decode=False,
            temperature=temperature,
        )

        cyc_rec_loss = loss_fn(cyc_log_probs.transpose(1, 2), inp_tokens) * token_mask # 실제 input과 cycle x 비교 
        cyc_rec_loss = cyc_rec_loss.sum() / batch_size
        cyc_rec_loss *= config.cyc_factor

    # style consistency loss discriminator를 이용한 adversary loss
    else: 
        cyc_rec_loss = torch.tensor([0], device = config.device)
    adv_log_porbs = model_D(gen_soft_tokens, gen_lengths, rev_styles) # cycle시 만든 f(x, s_hat)을 discriminator에 삽입 shape (batch_size, num_styles) or (batch_size, 1)
    if config.discriminator_method == 'Multi': # Multi-class discriminator일 경우
        adv_labels = rev_styles + 1 # discriminator가 속아야 하기 때문에, x_hat이 문장이지만 s로 예측하도록 함. 
    else:
        adv_labels = torch.ones_like(rev_styles) # conditional discriminaotr일 경우 g(f(x, s_hat), s_hat)을 계산, discriminator가 속았다면 1(일치)를 return해야 함.
    adv_loss = loss_fn(adv_log_porbs, adv_labels) # loss 계산
    adv_loss = adv_loss.sum() / batch_size
    adv_loss *= config.adv_factor
        
    (cyc_rec_loss + adv_loss).backward() # 역전파
        
    # update parameters
    
    clip_grad_norm_(model_F.parameters(), 5) # gradient clipping 사용 max_norm : 5
    optimizer_F.step() # update

    model_D.train() 

    return slf_rec_loss.item(), cyc_rec_loss.item(), adv_loss.item()

def train(config, vocab, model_F, model_D, train_iters, dev_iters, num_epoch): # 실제 학습시키는 코드 
    optimizer_F = optim.Adam(model_F.parameters(), lr=config.pretrain_lr, weight_decay=config.L2) 
    optimizer_D = optim.Adam(model_D.parameters(), lr=config.lr_D, weight_decay=config.L2)

    his_d_adv_loss = [] # discriminator loss list
    his_f_slf_loss = []
    his_f_cyc_loss = []
    his_f_adv_loss = []
    
    #writer = SummaryWriter(config.log_dir)
    global_step = 0
    model_F.train()
    model_D.train()

    config.save_folder = config.save_path + '/' + str(time.strftime('%b%d%H%M%S', time.localtime()))
    os.makedirs(config.save_folder)
    os.makedirs(config.save_folder + '/ckpts')
    log_loss_dir = config.save_folder + '/loss_log.txt'

    print('Save Path:', config.save_folder)

    print('Model F pretraining......')
    for i, batch in enumerate(train_iters):
        if i >= config.F_pretrain_iter:
            break
        slf_loss, cyc_loss, _ = f_step(config, vocab, model_F, model_D, optimizer_F, batch, 1.0, 1.0, False)
        his_f_slf_loss.append(slf_loss)
        his_f_cyc_loss.append(cyc_loss)

        if (i + 1) % 10 == 0:
            avrg_f_slf_loss = np.mean(his_f_slf_loss)
            avrg_f_cyc_loss = np.mean(his_f_cyc_loss)
            his_f_slf_loss = []
            his_f_cyc_loss = []
            print('[iter: {}] slf_loss:{:.4f}, rec_loss:{:.4f}'.format(i + 1, avrg_f_slf_loss, avrg_f_cyc_loss))

            with open(log_loss_dir, "a") as f:
              f.write(
                '[iter: {}] slf_loss:{:.4f}, rec_loss:{:.4f} \n'.format(i + 1, avrg_f_slf_loss, avrg_f_cyc_loss)
              )
    
    print('Training start......')
    optimizer_F = optim.Adam(model_F.parameters(), lr=config.lr_F, weight_decay=config.L2) 
    def calc_temperature(temperature_config, step): # temperature도 dynamic progammed
        num = len(temperature_config)
        for i in range(num):
            t_a, s_a = temperature_config[i]
            if i == num - 1:
                return t_a
            t_b, s_b = temperature_config[i + 1]
            if s_a <= step < s_b:
                k = (step - s_a) / (s_b - s_a)
                temperature = (1 - k) * t_a + k * t_b
                return temperature
    for epoch in range(num_epoch):
        batch_iters = iter(train_iters)
        while True:
            drop_decay = calc_temperature(config.drop_rate_config, global_step)
            temperature = calc_temperature(config.temperature_config, global_step)
            batch = next(batch_iters)
            
            for _ in range(config.iter_D):
                batch = next(batch_iters)
                d_adv_loss = d_step(
                    config, vocab, model_F, model_D, optimizer_D, batch, temperature
                )
                his_d_adv_loss.append(d_adv_loss)
                
            for _ in range(config.iter_F):
                batch = next(batch_iters)
                if global_step < config.cyc_rec_loss_iter:
                    cyc_rec = False
                else:
                    cyc_rec = True
                f_slf_loss, f_cyc_loss, f_adv_loss = f_step(
                    config, vocab, model_F, model_D, optimizer_F, batch, temperature, drop_decay, cyc_rec = cyc_rec
                )
                his_f_slf_loss.append(f_slf_loss)
                his_f_cyc_loss.append(f_cyc_loss)
                his_f_adv_loss.append(f_adv_loss)
                
            
            global_step += 1
            #writer.add_scalar('rec_loss', rec_loss.item(), global_step)
            #writer.add_scalar('loss', loss.item(), global_step)
                
                
            if global_step % config.log_steps == 0:
                avrg_d_adv_loss = np.mean(his_d_adv_loss)
                avrg_f_slf_loss = np.mean(his_f_slf_loss)
                avrg_f_cyc_loss = np.mean(his_f_cyc_loss)
                avrg_f_adv_loss = np.mean(his_f_adv_loss)
                log_str = '[epoch : {} iter {:03}]d_adv_loss: {:.4f}  ' + \
                        'f_slf_loss: {:.4f}  f_cyc_loss: {:.4f}  ' + \
                        'f_adv_loss: {:.4f}  temp: {:.4f}  drop: {:.4f} \n'
                print(log_str.format(
                    epoch, global_step, avrg_d_adv_loss,
                    avrg_f_slf_loss, avrg_f_cyc_loss, avrg_f_adv_loss,
                    temperature, config.inp_drop_prob * drop_decay
                ))
                
                with open(log_loss_dir, "a") as f:
                  f.write(log_str.format(
                    epoch, global_step, avrg_d_adv_loss,
                    avrg_f_slf_loss, avrg_f_cyc_loss, avrg_f_adv_loss,
                    temperature, config.inp_drop_prob * drop_decay
                  ))
                
            if global_step % config.eval_steps == 0:
                his_d_adv_loss = []
                his_f_slf_loss = []
                his_f_cyc_loss = []
                his_f_adv_loss = []
                
                #save model
                torch.save(model_F.state_dict(), config.save_folder + '/ckpts/' + str(global_step) + '_F.pth')
                torch.save(model_D.state_dict(), config.save_folder + '/ckpts/' + str(global_step) + '_D.pth')
                auto_eval(config, vocab, model_F, dev_iters, global_step, temperature)

# 호출: auto_eval(config, vocab, model_F, dev_iters, global_step, temperature)
def auto_eval(config, vocab, model_F, test_iters, global_step, temperature):
    model_F.eval()
    vocab_size = len(vocab)
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
   
    evaluator = Evaluator()
    # print("----- start calculating accuaracy -----")
    try:
      acc_nonnative = evaluator.native_acc_0(rev_output[0])
      acc_native = evaluator.native_acc_1(rev_output[1])
    # print(f"nonnative accuracy : {acc_nonnative:.3f} ||||| native accuracy : {acc_native:.3f}")

    ####################################
    # print("----- start calculating bleu score -----")
      bleu_nonnative = evaluator.self_bleu_b(gold_text[0], rev_output[0])
      bleu_native = evaluator.self_bleu_b(gold_text[1], rev_output[1])
    # print(f"nonnavie bleu score : {bleu_nonnative:.3f} ||||| native bleu score : {bleu_native:.3f}")
    ####################################    
    
    # print("----- start calculating perplexity -----")
      ppl_nonnative = evaluator.native_ppl(rev_output[0])
      ppl_native = evaluator.native_ppl(rev_output[1])
      # print(f"nonnative perplexity : {ppl_nonnative:.3f} ||||| native perplexity : {ppl_native:.3f}")
    except:
      print("somethings wrong happen during evaluation")

   


    for k in range(5):
        idx = np.random.randint(len(rev_output[0]))
        print('*' * 20, 'nonnative sample', '*' * 20)
        print('[gold]', gold_text[0][idx])
        print('[raw ]', raw_output[0][idx])
        print('[rev ]', rev_output[0][idx])
    print('*' * 20, '********', '*' * 20)


    for k in range(5):
        idx = np.random.randint(len(rev_output[1]))
        print('*' * 20, 'native sample', '*' * 20)
        print('[gold]', gold_text[1][idx])
        print('[raw ]', raw_output[1][idx])
        print('[rev ]', rev_output[1][idx])

    print('*' * 20, '********', '*' * 20)

    print(('[auto_eval] acc_native: {:.4f} acc_nonnative: {:.4f} ' + \
          'bleu_native: {:.4f} bleu_nonnative: {:.4f} ' + \
          'ppl_native: {:.4f} ppl_nonnative: {:.4f}\n').format(
              acc_native, acc_nonnative, bleu_native, bleu_nonnative, ppl_native, ppl_nonnative,
    ))


    # save output
    save_file = config.save_folder + '/' + str(global_step) + '.txt'
    eval_log_file = config.save_folder + '/eval_log.txt'
    with open(eval_log_file, 'a') as fl:
        print(('iter{:5d}:  acc_native: {:.4f} acc_nonnative: {:.4f} ' + \
               'bleu_native: {:.4f} bleu_nonnative: {:.4f} ' + \
               'ppl_native: {:.4f} ppl_nonnative: {:.4f}\n').format(
            global_step, acc_native, acc_nonnative, bleu_native, bleu_nonnative, ppl_native, ppl_nonnative,
        ), file=fl)
    with open(save_file, 'w') as fw:
        print(('[auto_eval] acc_native: {:.4f} acc_nonnative: {:.4f} ' + \
               'bleu_native: {:.4f} bleu_nonnative: {:.4f} ' + \
               'ppl_native: {:.4f} ppl_nonnative: {:.4f}\n').format(
            acc_native, acc_nonnative, bleu_native, bleu_nonnative, ppl_native, ppl_nonnative,
        ), file=fw)

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
