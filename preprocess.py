import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.utils.data
from torchtext.vocab import vocab
import torchtext.transforms as T
from janome.tokenizer import Tokenizer
import spacy
from collections import Counter
import os
import pickle

from pandarallel import pandarallel
pandarallel.initialize()

def get_dataset_and_vocab(src_max_len,tgt_max_len,source_vocab_max_size,target_vocab_max_size,dataset_name):
    j_word_count = src_max_len
    e_word_count = tgt_max_len
    #日本語用のトークン変換関数を作成
    global j_t # 並列処理のため
    global e_t # 並列処理のため 5枚以上程度早くなる気がする。
    j_t = Tokenizer()
    def j_tokenizer(text): 
        return [tok for tok in j_t.tokenize(text, wakati=True)]

    #英語用のトークン変換関数を作成
    e_t = spacy.load('en_core_web_sm')
    def e_tokenizer(text):
        return [tok.text for tok in e_t.tokenizer(text)]

    print("data load start")
    if dataset_name=="kftt":
        pref="/mnt/hdd/dataset/kftt-data-1.0/data/orig"
        with open(os.path.join(pref,"kyoto-train.ja"),"r") as f:
            ja_lines=f.readlines()
        ja_lines=list(map(lambda x:x.strip(),ja_lines))
        with open(os.path.join(pref,"kyoto-train.en"),"r") as f:
            en_lines=f.readlines()
        en_lines=list(map(lambda x:x.strip(),en_lines))
    elif dataset_name=="JEC":
        df = pd.read_excel("./JEC_basic_sentence_v1-3.xls", header = None)
        ja_lines=df.iloc[:,1].to_list()
        en_lines=df.iloc[:,2].to_list()
    print("data loaded")
    data=list(zip(ja_lines,en_lines))
    df=pd.DataFrame(data,columns=["src_token","tgt_token"])


    #各文章をトークンに変換　重い。
    print("tokenize")
    src_pikle=f"{dataset_name}_src.pickle"
    tgt_pikle=f"{dataset_name}_tgt.pickle"
    if os.path.isfile(src_pikle):
        with open(src_pikle, 'rb') as f:
            texts = pickle.load(f)
    else:
        texts = df["src_token"].parallel_apply(j_tokenizer) # origin apply
        with open(src_pikle, 'wb') as f:
            pickle.dump(texts, f)
    if os.path.isfile(tgt_pikle):
        with open(tgt_pikle, 'rb') as f:
            targets = pickle.load(f)
    else:
        targets = df["tgt_token"].parallel_apply(e_tokenizer) # origin apply
        with open(tgt_pikle, 'wb') as f:
            pickle.dump(targets, f)
    print("tokenize end")
    print(texts[0])
    print(targets[0])


    #日本語のトークン数（単語数）をカウント
    j_list = []
    for i in range(len(texts)):
        j_list.extend(texts[i])
    j_counter=Counter()
    j_counter.update(j_list)
    specials=['<unk>', '<pad>', '<bos>', '<eos>']
    j_counter=Counter(dict(j_counter.most_common(source_vocab_max_size-len(specials))))
    j_v = vocab(j_counter, specials=(specials))   #特殊文字の定義
    j_v.set_default_index(j_v['<unk>'])
    print("j_vocab created")
    print("j_v",len(j_v))

    #英語のトークン数（単語数）をカウント
    e_list = []
    for i in range(len(targets)):
        e_list.extend(targets[i])
    e_counter=Counter()
    e_counter.update(e_list)
    specials=['<unk>', '<pad>', '<bos>', '<eos>']
    e_counter=Counter(dict(e_counter.most_common(target_vocab_max_size-len(specials))))
    e_v = vocab(e_counter,specials=(specials))   #特殊文字の定義
    e_v.set_default_index(e_v['<unk>'])
    print("en_vocab created")
    print("en_v",len(e_v))


    j_text_transform = T.Sequential(
    T.VocabTransform(j_v),   #トークンに変換
    T.Truncate(j_word_count),   #14語以上の文章を14語で切る
    T.AddToken(token=j_v['<bos>'], begin=True),   #文頭に'<bos>
    T.AddToken(token=j_v['<eos>'], begin=False),   #文末に'<eos>'を追加
    T.ToTensor(),   #テンソルに変換
    T.PadTransform(j_word_count + 2, j_v['<pad>'])   #14語に満たない文章を'<pad>'で埋めて14語に合わせる
    )

    e_text_transform = T.Sequential(
    T.VocabTransform(e_v),   #トークンに変換
    T.Truncate(e_word_count),   #14語以上の文章を14語で切る
    T.AddToken(token=e_v['<bos>'], begin=True),   #文頭に'<bos>
    T.AddToken(token=e_v['<eos>'], begin=False),   #文末に'<eos>'を追加
    T.ToTensor(),   #テンソルに変換
    T.PadTransform(e_word_count + 2, e_v['<pad>'])   #14語に満たない文章を'<pad>'で埋めて14語に合わせる
    )

    class Dataset(torch.utils.data.Dataset):
        def __init__(
            self,
            texts,targets,
            j_text_transform,
            e_text_transform,
            ):
            
            self.texts=texts # src
            self.targets=targets # 対応する正解の分割された配列
            # self.texts = df["src_token"].parallel_apply(j_tokenizer)
            # self.targets = df["tgt_token"].parallel_apply(e_tokenizer)
            self.j_text_transform = j_text_transform
            self.e_text_transform = e_text_transform
        
        def max_word(self):
            return len(self.j_v), len(self.e_v)
                
        def __getitem__(self, i):
            text = self.texts[i]
            text = self.j_text_transform([text]).squeeze()

            target = self.targets[i]
            target = self.e_text_transform([target]).squeeze()
            data = {"src_token": text, "tgt_token": target}
            return data
        
        def __len__(self):
            return len(self.texts)
    dataset = Dataset(texts,targets, j_text_transform, e_text_transform)
    print("preprocess ended")
    return dataset , j_v, e_v , j_tokenizer,e_tokenizer