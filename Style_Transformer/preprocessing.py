# !pip install chardet
import os
import re
import nltk 
import chardet
import pandas as pd
import numpy as np
from random import shuffle
from collections import Counter
from bs4 import BeautifulSoup
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
nltk.download('punkt')


def preprocess_native_transformer(
    yelc_path = "./data/dataset/Raw/YELC_2011/",
    use_path = "./data/dataset/Raw/USEcorpus/",
    academic_path = "./data/dataset/Raw/COCA/text_acad.txt",
    save_path = "./data/dataset/classifier_data", 
    native_path = "./data/dataset",
    dataset_ratio = (0.7, 0.2, 0.1),
    use_discriminator = True):
    #save_path : native 데이터 전처리 후 저장 폴더 
    # 1. YELC 전처리
    yelc_level = pd.read_csv( yelc_path+"YELC_2011.csv" )

    nonnative_id = yelc_level.loc[(yelc_level.Grade=='A1')|(yelc_level.Grade=='A1+')|(yelc_level.Grade=='A2'),'Student ID']
    native_id = yelc_level.loc[(yelc_level.Grade=='C1')|(yelc_level.Grade=='C2'),'Student ID']
    yelc_nonn_files = [str(file) + num + '.txt' for file in nonnative_id for num in ['_01','_02']]
    yelc_native_files = [str(file) + num + '.txt' for file in native_id for num in ['_01','_02']]

    yelc_nonnative = []
    for yelc in yelc_nonn_files:
      with open(yelc_path + 'YELC_2011/' + yelc, encoding = 'UTF-8', newline = '\n',errors='ignore') as f:
        corpus = f.readlines()
        preprocessed = sum([preprocessing(sent) for sent in corpus], [])
        yelc_nonnative.extend(preprocessed)

    yelc_native = []
    for yelc in yelc_native_files:
      with open(yelc_path + 'YELC_2011/' + yelc, encoding = 'UTF-8', newline = '\n',errors='ignore') as f:
        corpus = f.readlines()
        preprocessed = sum([preprocessing(sent) for sent in corpus], [])
        yelc_native.extend(preprocessed)

    # 2. USE 전처리
    use_files = os.listdir(use_path)
    use_native = []
    for level in use_files:
      if 'c' in level:
          file_list = os.listdir(use_path + level)
          for file in file_list:
            try:
                text = load_use(use_path + level + "/" + file)
            except :
                print("error file name : ", file)
            use_native.extend(text)
    use_native = [text.replace("\t", "") for text in use_native]
    use_native = [text for text in use_native if 'title' not in text]

    use_nonnative = []
    for level in use_files:
      if "a" in level:
          file_list = os.listdir(use_path + level)
          for file in file_list:
            try: 
                text = load_use(use_path + level + '/' + file)
            except : 
                print("error file name : ", file)
          use_nonnative.extend(text)
    use_nonnative = [text.replace('\t', '') for text in use_nonnative]
    use_nonnative = [text for text in use_nonnative if 'title' not in text]

    academic = pd.read_csv(academic_path, delimiter = '\t', header = None)[0].tolist()

    native = yelc_native + use_native  + academic
    nonnative = yelc_nonnative + use_nonnative

    print("-------------- Before Preprocessing ----------------")
    print(f"num of sentence of native : {len(native)}, num of sentence of nonnative : {len(nonnative)}")
    print("----------------------------------------------------")
    native_preprocessing = [preprocessing(line) for line in native]
    nonnative_preprocessing = [preprocessing(line) for line in nonnative]

    native_preprocessing = sum(native_preprocessing, [])
    nonnative_preprocessing = sum(nonnative_preprocessing, [])

    native_preprocessing = [text for text in native_preprocessing if (5 < len(word_tokenize(text)) < 32)]
    nonnative_preprocessing = [text for text in nonnative_preprocessing if (5 < len(word_tokenize(text)) < 32)]
    print("-------------- After Preprocessing ----------------")
    print(f"num of sentence of native : {len(native_preprocessing)}, num of sentence of nonnative : {len(nonnative_preprocessing)}")
    print("----------------------------------------------------")

    shuffle(native_preprocessing)
    shuffle(nonnative_preprocessing)

    train_ratio, val_ratio, test_ratio = dataset_ratio

    train_len_native, val_len_native = int(len(native_preprocessing)*train_ratio), int(len(native_preprocessing)*val_ratio)
    train_len_nonnative, val_len_nonnative = int(len(nonnative_preprocessing)*train_ratio), int(len(nonnative_preprocessing)*val_ratio)

    train_native = native_preprocessing[:train_len_native]
    val_native = native_preprocessing[train_len_native:val_len_native]
    test_native = native_preprocessing[val_len_native:]

    train_nonnative = nonnative_preprocessing[:train_len_nonnative]
    val_nonnative = nonnative_preprocessing[train_len_nonnative:val_len_nonnative]
    test_nonnative = nonnative_preprocessing[val_len_nonnative:]
    
    if use_discriminator:
      train_native_pd = pd.DataFrame(list(zip(train_native, [0]*len(train_native))), columns = ['text', 'label'])
      val_native_pd = pd.DataFrame(list(zip(val_native, [0]*len(val_native))), columns = ['text', 'label'])
      test_native_pd = pd.DataFrame(list(zip(test_native, [0]*len(test_native))), columns = ['text', 'label'])

      train_nonnative_pd = pd.DataFrame(list(zip(train_nonnative, [0]*len(train_nonnative))), columns = ['text', 'label'])
      val_nonnative_pd = pd.DataFrame(list(zip(val_nonnative, [0]*len(val_nonnative))), columns = ['text', 'label'])
      test_nonnative_pd = pd.DataFrame(list(zip(test_nonnative, [0]*len(test_nonnative))), columns = ['text', 'label'])

      train_df = pd.concat([train_native_pd, train_nonnative_pd])
      train_df.to_csv(save_path + '/train.csv', index = False)

      val_df = pd.concat([val_native_pd, val_nonnative_pd])
      val_df.to_csv(save_path + '/val.csv', index = False)

      test_df = pd.concat([test_native_pd, test_nonnative_pd])
      test_df.to_csv(save_path + '/test.csv', index = False)

      native_pd = pd.DataFrame(native_preprocessing)
      nonnative_pd = pd.DataFrame(nonnative_preprocessing)

      native_pd.to_csv(save_path + '/native_preprocessed.csv', index = None, header = None)

      nonnative_pd.to_csv(save_path + '/nonnative_preprocessed.csv', index = None, header = None)

    else: # discriminator를 이용해 극성이 강한 데이터만 뽑지 않을 경우 그대로 native transformer 학습에 사용.   
      train_native_pd = pd.DataFrame(train_native)
      val_native_pd = pd.DataFrame(val_native)
      test_native_pd = pd.DataFrame(test_native)

      train_nonnative_pd = pd.DataFrame(train_nonnative)
      val_nonnative_pd = pd.DataFrame(val_nonnative)
      test_nonnative_pd = pd.DataFrame(test_nonnative)

      train_native_pd.to_csv(native_path + '/train_native.csv', index = None, header = None)
      val_native_pd.to_csv(native_path + '/val_native.csv', index = None, header = None)
      test_native_pd.to_csv(native_path + '/test_native.csv', index = None, header = None)

      train_nonnative_pd.to_csv(native_path + '/train_nonnative.csv', index = None, header = None)
      val_nonnative_pd.to_csv(native_path + '/val_nonnative.csv', index = None, header = None)
      test_nonnative_pd.to_csv(native_path + '/test_nonnative.csv', index = None, header = None)    

def preprocessing(corpus):
  corpus = BeautifulSoup(corpus, 'lxml').text
  corpus = re.sub("[\(\[].*?[\)\]]", '', corpus)
  corpus = re.sub("&nbsp;", " ", corpus)
  corpus = re.sub("&It;", "<", corpus)
  corpus = re.sub("%amp;", "and", corpus)
  corpus = re.sub("quot;", '"', corpus)
  corpus = re.sub("&#035;", "#", corpus)
  corpus = re.sub("&#039;", "'", corpus)

  # corpus = re.sub("\\'(.)", "'\\1", corpus)
  corpus = re.sub("wo n't", "won't", corpus)
  corpus = re.sub("(\$) ?([0-9])", "dollar \\2", corpus)
  corpus = re.sub("%", 'percent', corpus)
  corpus = re.sub("@!", "", corpus)
  corpus = re.sub("^[0-9a-zA-Z]([-_.]?[0-9a-zA-Z])*@[0-9a-zA-Z]([-_.]?[0-9a-zA-Z])*.[a-zA-Z]{2,3}$/i", "EMAIL", corpus) # 이메일
  corpus = re.sub('http : //\S+|https : //\S+', '', corpus, flags=re.MULTILINE) # not complete
  corpus = re.sub("[\(\[] ?.*? ?[\)\]]", '', corpus) 
  corpus = re.sub("[0-9]{1,}", " NUM ", corpus) 

  corpus = re.sub('\.', 'thisIsSpecialTokenForProd', corpus)
  corpus = re.sub(" ", "thisIsSpecialTokenForBlank", corpus)
  corpus = re.sub("(\W){2,}", "\\1", corpus)
  corpus = re.sub("thisIsSpecialTokenForProd", ".", corpus)
  corpus = re.sub("thisIsSpecialTokenForBlank", " ", corpus)
  
  corpus = re.sub("\.{2,}", ".", corpus)
  corpus = sent_tokenize(corpus)
  corpus = [sentence for sentence in corpus if (len(sentence.split(" ")) > 5) and ("@ @ @ @ @" not in sentence)]
  return corpus


def decode_text(text):
  try:
    result = text.decode('ascii')
  except:
    result = text.decode(chardet.detect(text)['encoding'])
  return result

def load_use(file_path):
  try:
    with open(file_path) as f:
      test = f.readlines()
      test = [text.replace("\n", "") for text in test]
  except:
    with open(file_path, 'rb') as f:
      test = f.readlines()
      test = [decode_text(text) for text in test]
      test = [text.replace('\r\n', '') for text in test]
  test = [text for text in test if len(text) > 40]
  return test

if __name__ == "__main__":
  print("preprocessing start.")
  preprocess_native_transformer()
  print("Preprocessing Native Transformer Dataset is Over.")