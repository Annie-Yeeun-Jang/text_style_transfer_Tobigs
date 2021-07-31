from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM

import fasttext
import pkg_resources
import math
from numpy import mean
import torch
from torch.nn import Softmax

class Evaluator(object):

    def __init__(self):
        resource_package = __name__

        native_fasttext = 'native_fasttext.bin'

        native_fasttext_file = pkg_resources.resource_stream(resource_package, native_fasttext)

        self.classifier_native = fasttext.load_model(native_fasttext_file.name)
        self.native_ppl_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        
        self.smoothing = SmoothingFunction().method4

    # acc_b에서 사용
    def native_style_check(self, text_transfered, style_origin):
        text_transfered = ' '.join(word_tokenize(text_transfered.lower().strip()))
        if text_transfered == '':
            return False
        label = self.classifier_native.predict([text_transfered])
        style_transfered = label[0][0] == '__label__positive'
        return (style_transfered != style_origin)

    # acc 측정 위한 함수 (지금은 생략함)
    def native_acc_b(self, texts, styles_origin):
        assert len(texts) == len(styles_origin), 'Size of inputs does not match!'
        count = 0
        for text, style in zip(texts, styles_origin):
            if self.native_style_check(text, style):
                count += 1
        return count / len(texts)

    def native_acc_0(self, texts):
        styles_origin = [0] * len(texts)
        return self.native_acc_b(texts, styles_origin)

    def native_acc_1(self, texts):
        styles_origin = [1] * len(texts)
        return self.native_acc_b(texts, styles_origin)

    # BLEU 측정 위한 함수 (지금은 생략함)
    def nltk_bleu(self, texts_origin, text_transfered):
        texts_origin = [word_tokenize(text_origin.lower().strip()) for text_origin in texts_origin]
        text_transfered = word_tokenize(text_transfered.lower().strip())
        return sentence_bleu(texts_origin, text_transfered, smoothing_function = self.smoothing) * 100
    
    def self_bleu_b(self, texts_origin, texts_transfered):
        assert len(texts_origin) == len(texts_transfered), 'Size of inputs does not match!'
        sum = 0
        n = len(texts_origin)
        for x, y in zip(texts_origin, texts_transfered):
            try : 
                bleu = self.nltk_bleu([x], y)
            except ZeroDivisionError:
                bleu = 0
            sum += bleu
        return sum / n

    # ppl 체크를 위한 함수
    def native_ppl(self, texts_transfered): #생성된 문장이 input
      softmax = Softmax(dim = 0)
      tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      tokenize_input = [tokenizer.tokenize(line) for line in texts_transfered]
      tensor_input = [torch.tensor(tokenizer.convert_tokens_to_ids(line)).unsqueeze_(1).to(torch.int64) for line in tokenize_input]
      
      ppl_result = []
      for sentence in tensor_input:
        sentence_prediction = self.native_ppl_model(sentence)
        sentence_confidence = softmax(sentence_prediction).squeeze_(dim = 1)
        sentence_ppl_list = [confidence[token_idx].item() for confidence, token_idx in zip(sentence_confidence, sentence)] 
        length = len(sentence_ppl_list)
        if length == 0 : length = 1 
        sentence_ppl = prod_list(sentence_ppl_list)**(-1/length)

        ppl_result.append(sentence_ppl)

      return mean(ppl_result)
    
def prod_list(ppl_list):
    result = 1
    for elem in ppl_list:
        result *= elem
    return result

