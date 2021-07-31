"""
    data_path : the path of the datasets
    log_dir : where to save the logging info
    save_path = where to save the checkpoing
    
    discriminator_method : the type of discriminator ('Multi' or 'Cond')
    min_freq : the minimun frequency for building vocabulary
    max_length : the maximun sentence length 
    embed_size : the dimention of the token embedding
    d_model : the dimention of Transformer d_model parameter
    h : the number of Transformer attention head
    num_layers : the number of Transformer layer
    batch_size : the training batch size
    lr_F : the learning rate for the Style Transformer
    lr_D : the learning rate for the discriminator
    L2 : the L2 norm regularization factor
    iter_D : the number of the discriminator update step pre training interation
    iter_F : the number of the Style Transformer update step pre training interation
    dropout : the dropout factor for the whole model

    log_steps : the number of steps to log model info
    eval_steps : the number of steps to evaluate model info

    slf_factor : the weight factor for the self reconstruction loss
    cyc_factor : the weight factor for the cycle reconstruction loss
    adv_factor : the weight factor for the style controlling loss

    incremental --- : recursive한 forward
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from utils import idx2onehot


class StyleTransformer(nn.Module):
    def __init__(self, config, vocab):
        super(StyleTransformer, self).__init__()
        num_styles, num_layers = config.num_styles, config.num_layers  # 스타일 개수, 레이어 개수 설정
        d_model, max_length = config.d_model, config.max_length  # 내부 임베딩 벡터, 문장 길이 결정
        h, dropout = config.h, config.dropout  # h는 attention head 개수 dropout ratio 결정
        learned_pos_embed = config.learned_pos_embed  # positional embedding 가져오기
        load_pretrained_embed = config.load_pretrained_embed  # 사전 학습된 임베딩 가져오기
        
        self.d_embed = config.embed_size
        self.max_length = config.max_length  # 문장 최대 길이 다시 가져온다...
        self.eos_idx = vocab.stoi['<eos>']  # eos 토큰 인덱스
        self.pad_idx = vocab.stoi['<pad>']  # pad 토큰 인덱스
        self.style_embed = Embedding(num_styles, self.d_embed)  # 스타일 임베딩 레이어 만들기 (style 수, 차원수)
        self.embed = EmbeddingLayer(  # 문장 임베딩 레이어 만들기
            vocab, self.d_embed, max_length,
            self.pad_idx,
            learned_pos_embed,
            load_pretrained_embed,
        )
        self.sos_token = nn.Parameter(torch.randn(d_model))  # sos 토큰 벡터 shape : (d_model)
        self.encoder = Encoder(num_layers, self.d_embed, d_model, len(vocab), h, dropout)  # 인코더 단 만들기
        self.decoder = Decoder(num_layers, self.d_embed, d_model, len(vocab), h, dropout)  # 디코더 단 만들기

    def forward(self, inp_tokens, gold_tokens, inp_lengths, style,
                # inp_tokens : src sentences, gold_tokens : src sentences used in decoder
                generate=False, differentiable_decode=False, temperature=1.0):
        batch_size = inp_tokens.size(0)  # 입력 데이터 row 수 == 배치 사이즈
        max_enc_len = inp_tokens.size(1)  # 입력 데이터 col 수 == 이번 배치의 max len

        assert max_enc_len <= self.max_length  # 실제 max len보다 이번 배치의 max len이 짧은지 확인

        pos_idx = torch.arange(self.max_length).unsqueeze(0).expand((batch_size, -1))  # max len만큼의 벡터를 만들고, (1, max_len)으로 모양 변경 후 (batch_size, max_len)으로 확장
        pos_idx = pos_idx.to(inp_lengths.device)  # gpu에 올림(inp_lengths는 이미 올라가 있는 상황)

        src_mask = pos_idx[:, :max_enc_len] >= inp_lengths.unsqueeze(-1)  # 현재 배치의 max len이후의 토큰들은 False로된 행렬 생성
        src_mask = torch.cat((torch.zeros_like(src_mask[:, :1]), src_mask), 1)  # src_mask의 맨 앞 col에 0 토큰 부착(sos 토큰인가?)
        src_mask = src_mask.view(batch_size, 1, 1, max_enc_len + 1)  # 차원 수 변경, 위에서 0 토큰을 부착해서 max_enc_len + 1임

        tgt_mask = torch.ones((self.max_length, self.max_length)).to(src_mask.device)  # (max_len, max_len)의 행렬(1로 이루어짐) 생성 후 gpu 올림
        tgt_mask = (tgt_mask.tril() == 0).view(1, 1, self.max_length, self.max_length)  # tril : 아래 삼각행렬 반환 함수// 아래 삼각 행렬을 False로 만든 후 (1, 1, max_len, max_len) 모양으로 변형// 디코더는 masked attetion이라 그런듯

        style_emb = self.style_embed(style).unsqueeze(1)  # transfer할 스타일 임베딩 가져와서 (d_model, 1)로 변경

        enc_input = torch.cat((style_emb, self.embed(inp_tokens, pos_idx[:, :max_enc_len])),1)  
            # self.embed를 통해 positional encoding된 embedding((batch_size, max_len, d_model)일듯) 을 가져오고, 이를 style embedding과 concat 이때 style embedding은 positional encoding 안함.
        memory = self.encoder(enc_input, src_mask)  # src 문장을 인코더에 통과시킴

        sos_token = self.sos_token.view(1, 1, -1).expand(batch_size, -1,-1)  # (d_model) -> (1, 1, d_model) -> (batch_size, 1, d_model)

        if not generate:  # generate : 훈련에 쓰이는지, inference에 쓰이는지 차이인듯. 훈련시엔 전체 행렬이 한번에 처리되고, inference 시엔 incremental forward를 통해 recursive를 트릭으로 처리한다.
            dec_input = gold_tokens[:, :-1]  # gold_tokens의 마지막 단어는 쓰지 않는다. 당연하다.
            max_dec_len = gold_tokens.size(1)  # gold_tokens의 shape은 (batch_size, max_len)
            dec_input_emb = torch.cat((sos_token, self.embed(dec_input, pos_idx[:, :max_dec_len - 1])), 1)  
                # gold_tokens 맨앞에 sos token embedding을 붙여서 embedding한다. shape은 (batch_size, max_len, d_model) sos_token을 붙였기 때문에 max_len 유지
            log_probs = self.decoder(  # 디코더를 통과하여 최종적인 log probs된 토큰 값 출력. 이때 target mask는 크기를 잘라서 사용해서 행렬 연산을 줄인다.
                dec_input_emb, memory,
                src_mask, tgt_mask[:, :, :max_dec_len, :max_dec_len],
                temperature
            )
        else:

            log_probs = []
            next_token = sos_token  # 첫번째 토큰은 당연히 sos
            prev_states = None  # 아직은 없다. recursive하면서 전달되는 인자.

            for k in range(self.max_length):  # 이전에 예측한 softmax dist.를 이용해 soft embedding을 만들고 이번 시점의 입력값으로 사용한다.
                log_prob, prev_states = self.decoder.incremental_forward(  # 디코더를 통과하여 이번 시점의 softmax dist. 와 state 생성
                    next_token, memory,
                    src_mask, tgt_mask[:, :, k:k + 1, :k + 1],
                    temperature,
                    prev_states
                )

                log_probs.append(
                    log_prob)  # 각 시점의 softmax dist.를 저장 shape : (seq_len, vocab_size)인데, 실제론 리스트로 묶여있기 때문에 아님.

                if differentiable_decode:
                    next_token = self.embed(log_prob.exp(), pos_idx[:, k:k + 1])  # next_token은 softmax dist.를 가중합하여 생성
                else:
                    next_token = self.embed(log_prob.argmax(-1), pos_idx[:, k:k + 1])  # next_token은 현재 시점의 출력 토큰

                # if (pred_tokens == self.eos_idx).max(-1)[0].min(-1)[0].item() == 1:
                #    break

            log_probs = torch.cat(log_probs, 1)  # shape을 shape : (seq_len, vocab_size)로 만들어줌

        return log_probs


class Discriminator(nn.Module):  # 훈련시 사용될 판별자
    def __init__(self, config, vocab):
        super(Discriminator, self).__init__()
        num_styles, num_layers = config.num_styles, config.num_layers
        d_model, max_length = config.d_model, config.max_length
        h, dropout = config.h, config.dropout
        learned_pos_embed = config.learned_pos_embed
        load_pretrained_embed = config.load_pretrained_embed
        num_classes = config.num_classes  # 각종 하이퍼파라미터 및 옵션 설정

        self.d_embed = config.embed_size
        self.pad_idx = vocab.stoi['<pad>']  # pad 인덱스 저장 # stoi - token string / index dict
        self.style_embed = Embedding(num_styles, d_model)  # style embedding 생성
        self.embed = EmbeddingLayer(  # 토큰 임베딩 생성
            vocab, self.d_embed, max_length,
            self.pad_idx,
            learned_pos_embed,
            load_pretrained_embed
        )
        self.cls_token = nn.Parameter(torch.randn(d_model))  # cls token 생성
        self.encoder = Encoder(num_layers, self.d_embed, d_model, len(vocab), h, dropout)  # 모델의 인코더 구조 사용.
        self.classifier = Linear(d_model, num_classes)  # classifier로 단순 선형식 사용.

    def forward(self, inp_tokens, inp_lengths, style=None):  # 스타일이 주어지면 conditional, 주어지지 않으면 multi-class//
        # input_tokens : (batch_size, max_seq_len), inp_lengths : (batch_size), 문장별 문장 길이 벡터
        batch_size = inp_tokens.size(0)
        num_extra_token = 1 if style is None else 2  # style이 주어지지 않았다면, 생성된 문장을 의미하는 추가 클래스 하나, style이 주어졌다면, 스타일과 일치하는지 여부에 따라 클래스 두개.
        # 밑에서 스티알이 주어지지 않았다면 기존의 style_num을 이용해 class를 만든다.
        max_seq_len = inp_tokens.size(1)  # max_sea_len : 최대 문장 길이

        pos_idx = torch.arange(max_seq_len).unsqueeze(0).expand((batch_size, -1)).to(
            inp_lengths.device)  # max_seq_len의 벡터를 만들고, (1, max_seq_len)으로 차원을 바꾼뒤, (batch_size, max_seq_len)으로 반복해서 늘리고, gpu에 올림
        mask = pos_idx >= inp_lengths.unsqueeze(-1)  # 각 문장의 pad에 해당하는 mask 생성
        for _ in range(num_extra_token):
            mask = torch.cat((torch.zeros_like(mask[:, :1]), mask),1)  # cls 토큰이 들어갈 자리를 만들어서 mask 만들기 (batch_size, num_exta_token + max_seq_len)
        mask = mask.view(batch_size, 1, 1, max_seq_len + num_extra_token)

        cls_token = self.cls_token.view(1, 1, -1).expand(batch_size, -1, -1)  # (d_model) -> (batch_size, 1, d_model)

        enc_input = cls_token
        if style is not None:
            style_emb = self.style_embed(style).unsqueeze(1)  # (batch_size, d_model) -> (batch_size, 1, d_model) # 각 문장의 스타일
            enc_input = torch.cat((enc_input, style_emb), 1)  # (batch_size, 2, d_model)

        enc_input = torch.cat((enc_input, self.embed(inp_tokens, pos_idx)),1)  # (batch_size, num_extra_token + max_seq_len, d_model)

        encoded_features = self.encoder(enc_input, mask)
        logits = self.classifier(encoded_features[:, 0])

        return F.log_softmax(logits, -1)


class Encoder(nn.Module):  # 인코더 단
    def __init__(self, num_layers, d_embed, d_model, vocab_size, h, dropout):
        super(Encoder, self).__init__()
        EncoderLayerEmbed2Hidden = EncoderLayer(d_embed, h, dropout)
        EncoderLayerList = [EncoderLayer(d_model, h, dropout) for _ in range(num_layers - 1)]
        EncoderLayerList.insert(0, EncoderLayerEmbed2Hidden)
        self.layers = nn.ModuleList(EncoderLayerList)  # num_layers만큼 인코더를 쌓는다.
        self.norm = LayerNorm(d_model)  # 최종 output에 LayerNorm을 해준다.

    def forward(self, x, mask):
        y = x

        assert y.size(1) == mask.size(-1)  # mask와 입력 문장의 길이가 동일한지 확인

        for layer in self.layers:
            y = layer(y, mask)  # 레이어를 통과하며 지나감

        return self.norm(y)  # LayerNorm 후 return


class Decoder(nn.Module):  # 전체 디코더
    def __init__(self, num_layers, d_embed, d_model, vocab_size, h, dropout):
        super(Decoder, self).__init__()
        DecoderLayerEmbed2Hidden = DecoderLayer(d_embed, h, dropout)
        DecoderLayerList = [DecoderLayer(d_model, h, dropout) for _ in range(num_layers - 1)]
        DecoderLayerList.insert(0, DecoderLayerEmbed2Hidden)
        self.layers = nn.ModuleList(DecoderLayerList)  # 디코더 레이어 쌓기
        self.norm = LayerNorm(d_model)  # 디코더 최종 출력에도 LayerNorm을 해준다.
        self.generator = Generator(d_model, vocab_size)  # vocab 중에서 예측하니 당연히 vocab_size

    def forward(self, x, memory, src_mask, tgt_mask, temperature):  # memory : encoder에서 넘어온 K, V
        y = x

        assert y.size(1) == tgt_mask.size(
            -1)  # y : (batch_size, max_len, d_model) tgt_mask : (1, 1, max_len, max_len) # max_len이 같은지 확인

        for layer in self.layers:
            y = layer(y, memory, src_mask, tgt_mask)  # 레이어 순차적으로 통과

        return self.generator(self.norm(y), temperature)  # 마지막에 log softmax를 통과, 이때 temperature softmax를 함.

    def incremental_forward(self, x, memory, src_mask, tgt_mask, temperature,
                            prev_states=None):  # prev_states shape : (num_layers, (t-1), d_model)
        y = x

        new_states = []

        for i, layer in enumerate(self.layers):
            y, new_sub_states = layer.incremental_forward(
                y, memory, src_mask, tgt_mask,
                prev_states[i] if prev_states else None
            )

            new_states.append(new_sub_states)  # 레이어를 순차적으로 통과하면서 각 서브레이어의 states를 리스트로 저장

        new_states.append(torch.cat((prev_states[-1], y),
                                    1) if prev_states else y)  # 여기서 prev_states의 마지막은 마지막 디코더의 prev_states가 아니라 별도의 마지막 prev_states다. 여기엔 모든 레이어를 통과한 states가 저장되게 된다.
        y = self.norm(new_states[-1])[:, -1:]  # 현재 시점에 해당하는 attetion 결과만 뽑는다.

        return self.generator(y, temperature), new_states  # log softmax 후 반환


class Generator(nn.Module):  # 디코더 마지막에 붙는 최종 생성 레이어 softmax 레이어임
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, temperature):
        return F.log_softmax(self.proj(x) / temperature, dim=-1)


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab, d_embed, max_length, pad_idx, learned_pos_embed, load_pretrained_embed):
        super(EmbeddingLayer, self).__init__()
        self.token_embed = Embedding(len(vocab), d_embed)  # 문장을 토큰 별로 임베딩 하기 위한 레이어 생성 shape은 (len_vocab, d_model)
        self.pos_embed = Embedding(max_length, d_embed)  # postional encoding을 위한 레이어 생성
        self.vocab_size = len(vocab)
        if load_pretrained_embed:
            self.token_embed = nn.Embedding.from_pretrained(vocab.vectors)  # 사전 학습된 임베딩 벡터를 사용한 경우 그대로 가져와서 임베딩 레이어 생성
            print('embed loaded.')

    def forward(self, x, pos):  # x는 one hot vector의 concat 형태로 여기서 lookup하게 되거나, softmax dist.로 입력됨
        if len(x.size()) == 2:  # input이 token 번호로된 리스트인 경우
            y = self.token_embed(x) + self.pos_embed(pos)  # positional embedding과 token embedding을 그냥 더하면 됨
        else:  # input이 softmax dist.인 경우 shape : (1, vocab) token_embed.weight shape : (vocab, d_model)
            y = torch.matmul(x, self.token_embed.weight) + self.pos_embed(pos)  # matmul을 통해 해당 각각의 토큰과 dist를 곱하여 더하고 positional embedding을 더함.
        return y


class EncoderLayer(nn.Module):  # 인코더 단을 구성하는 서브레이어
    def __init__(self, d_model, h, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)  # self-attention을 수행하는 함수
        self.pw_ffn = PositionwiseFeedForward(d_model, dropout)  # FFNN을 수행하는 함수
        self.sublayer = nn.ModuleList(
            [SublayerConnection(d_model, dropout) for _ in range(2)])  # Residual connection과 LayerNorm을 수행하는 함수

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # 인코더 단은 self-attention만 수행한다.
        return self.sublayer[1](x, self.pw_ffn)  # FFNN 후 return


class DecoderLayer(nn.Module):  # 디코더를 만들기 위해 사용되는 디코더 1단
    def __init__(self, d_model, h, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)  # 디코더 블록의 첫 서브레이어는 masked self-attention. 여기서 사용할  self-attention 함수
        self.src_attn = MultiHeadAttention(d_model, h, dropout)  # 디코더 블록의 두번째 서브레이어는 인코더와의 attention 여기서 사용할 attention 함수
        self.pw_ffn = PositionwiseFeedForward(d_model, dropout)  # 디코더 블록의 마지막 서브레이어는 FFNN
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])  # 각 서브레이어들은 LayerNorm과 Residual Connection으로 구성. 이를 위해 연결할 블록 생성

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory  # 디코더에서 넘어오는 K, V
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # masked self-attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))  # attention
        return self.sublayer[2](x, self.pw_ffn)  # FFNN

    def incremental_forward(self, x, memory, src_mask, tgt_mask,
                            prev_states=None):  # recursive하게 출력하기 위해 prev_states가 출력되고 입력됨/// prev_states == 이전 시점의 attention 결과 (d_model, (t-1))
        new_states = []
        m = memory

        x = torch.cat((prev_states[0], x),
                      1) if prev_states else x  # prev_states가 없다 == 현재 첫 시점이다 == self attention에서 자신한테만 attentoin을 하면 된다. x의 shape : (d_model, 1(t)) // 즉 concat하여 (d_model, t)로 만든다.
        new_states.append(x)
        x = self.sublayer[0].incremental_forward(x, lambda x: self.self_attn(x[:, -1:], x, x,
                                                                             tgt_mask))  # query : x_t, k, v : x_1 ~ x_t
        x = torch.cat((prev_states[1], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[1].incremental_forward(x, lambda x: self.src_attn(x[:, -1:], m, m, src_mask))
        x = torch.cat((prev_states[2], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[2].incremental_forward(x, lambda x: self.pw_ffn(x[:, -1:]))
        return x, new_states  # new_states : 각 서브 레이어 별 attention 결과 


class MultiHeadAttention(nn.Module):  # transformer의 multihead attention과 똑같은 것 같다.
    def __init__(self, d_model, h, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0  # d_model이 attetion head 수로 나뉘는지 확인
        self.d_k = d_model // h  # 각 attention head의 차원 수 == d_k
        self.h = h
        self.head_projs = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)])  # Q, K, V를 만들기 위한 linear transformer
        self.fc = nn.Linear(d_model, d_model)  # head concat 후 linear transformer
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):  # 입력값은 사실 (x, x, x, mask)
        batch_size = query.size(0)  # q, k, v shape : (batch_size, max_len, d_model)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             # 입력 문장 x를 head projs에 통과시켜 (batch_size, num_head, max_len, d_k) head수만큼 나눠진 Q, K, V 만듦
                             for x, l in zip((query, key, value), self.head_projs)]

        attn_feature, _ = scaled_attention(query, key, value,
                                           mask)  # attn_feature : (batch_size, num_head, max_len, d_k) # attention 완료

        attn_concated = attn_feature.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                       self.h * self.d_k)  # attn_feature을 (batch_size, max_len, num_head, d_k)로 변경 후 새 주소값을 할당하고 (batch_size, max_len, d_model)로 변경하여 return

        return self.fc(attn_concated)


def scaled_attention(query, key, value, mask):
    d_k = query.size(-1)  # query shape : (batch_size, max_len, num_head, d_k)
    scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k)  # query : (batch_size, num_head, max_len, d_k)) key : (batch_size, num_head, d_k, max_len)를 matmul하여 attention score를 만든다. 이때 d_k로 sclaing
    scores.masked_fill_(mask, float('-inf'))  # score의 shape : (batch_size, num_head, max_len, max_len)기존의 masking된 부분은 그대로 making되도록 유지(-inf로)
    attn_weight = F.softmax(scores, -1)  # attention dist.로 변환
    attn_feature = attn_weight.matmul(value)  # attn_weight : (batch_size, num_head, max_len, max_len) value : (batch_size, num_head, max_len, d_k)
    # attn_features : (batch_size, num_head, max_len, d_k)
    return attn_feature, attn_weight


class PositionwiseFeedForward(nn.Module):  # 인코더와 디코더 블록의 마지막 서브레이어는 FFNN
    def __init__(self, d_model, dropout):  # 입력값의 shape : (batch_size, max_len, d_model)
        super(PositionwiseFeedForward, self).__init__()
        self.mlp = nn.Sequential(
            Linear(d_model, 4 * d_model),  # d_model -> 4*d_model로 맵핑
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            Linear(4 * d_model, d_model),  # 4*d_model -> d_model로 맵핑
        )

    def forward(self, x):
        return self.mlp(x)


class SublayerConnection(nn.Module):  # sublayer 연결 시 필요한 layer norm과 residual connection 구현
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):  # 출력에 layer norm을 하지 않고 input에 layernorm을 한다.
        y = sublayer(self.layer_norm(x))
        return x + self.dropout(y)  # residual connection

    def incremental_forward(self, x, sublayer):  # recursive하게 출력해야 할 경우
        y = sublayer(self.layer_norm(x))
        return x[:, -1:] + self.dropout(y)  # 이번 시점의 attention dist.만 얻으면 되기 때문에 마지막 시점 벡터인 -1만 출력하여 더해줌


def Linear(in_features, out_features, bias=True, uniform=True):
    m = nn.Linear(in_features, out_features, bias)
    if uniform:
        nn.init.xavier_uniform_(m.weight)
    else:
        nn.init.xavier_normal_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(embedding_dim, eps=1e-6):
    m = nn.LayerNorm(embedding_dim, eps)
    return m
