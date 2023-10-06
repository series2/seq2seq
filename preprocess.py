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
import json

from pandarallel import pandarallel
pandarallel.initialize()

# 現状はtokenizerなどは固定である。
def get_dataset_and_vocab(src_max_len,tgt_max_len,source_vocab_max_size,target_vocab_max_size,dataset_name):
    if not os.path.exists("preprocessed_data"):
        os.mkdir("preprocessed_data")
    prefix_path=f"preprocessed_data/{dataset_name}"
    if not os.path.exists(prefix_path):
        os.mkdir(prefix_path)
    info_dict={}
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

    print(f"data load start. name: `{dataset_name}`")
    if dataset_name=="kftt":
        pref="/mnt/hdd/dataset/kftt-data-1.0/data/orig"
        with open(os.path.join(pref,"kyoto-train.ja"),"r") as f:
            ja_lines=f.readlines()
        ja_lines=list(map(lambda x:x.strip(),ja_lines))
        with open(os.path.join(pref,"kyoto-train.en"),"r") as f:
            en_lines=f.readlines()
        en_lines=list(map(lambda x:x.strip(),en_lines))
    if dataset_name=="kftt_16k":
        pref="/mnt/hdd/dataset/kftt-data-1.0/data/orig"
        with open(os.path.join(pref,"kyoto-train.ja"),"r") as f:
            ja_lines=f.readlines()
        ja_lines=list(map(lambda x:x.strip(),ja_lines))
        with open(os.path.join(pref,"kyoto-train.en"),"r") as f:
            en_lines=f.readlines()
        en_lines=list(map(lambda x:x.strip(),en_lines))
    if dataset_name=="kftt_32k":
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
    elif dataset_name=="jesc":
        df=pd.read_table("/mnt/hdd/dataset/jesc/split/train",header=None,names=["en","ja"])
        ja_lines=df["ja"].to_list()
        en_lines=df["en"].to_list()
    else:
        raise Exception(f"dataset `{dataset_name}` not exists")

#  TODO 何かがおかしいのとみにくい
    plt.hist(list(map(len,ja_lines)))
    plt.xscale("log");plt.yscale("log")
    plt.savefig(f"{prefix_path}/src_token_dist.png")
    plt.hist(list(map(len,en_lines)))
    plt.xscale("log");plt.yscale("log")
    plt.savefig(f"{prefix_path}/tgt_token_dist.png")
    print("data loaded")
    data=list(zip(ja_lines,en_lines))
    df=pd.DataFrame(data,columns=["src_token","tgt_token"])


    #各文章をトークンに変換　重い。
    print("tokenize")
    src_pikle=f"{prefix_path}/src_tokens.pickle"
    tgt_pikle=f"{prefix_path}/tgt_tokens.pickle"
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
    info_dict.update(
        src_sentence_num=len(ja_lines),
        original_src_token_total_num=len(j_list),
        original_src_token_kind_num=len(j_counter)
    )
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
    info_dict.update(
        tgt_sentence_num=len(en_lines),
        original_tgt_token_total_num=len(e_list),
        original_tgt_token_kind_num=len(e_counter)
    )
    
    print("original tgt token 種類 : ",len(e_counter))
    specials=['<unk>', '<pad>', '<bos>', '<eos>']
    e_counter=Counter(dict(e_counter.most_common(target_vocab_max_size-len(specials))))
    e_v = vocab(e_counter,specials=(specials))   #特殊文字の定義
    e_v.set_default_index(e_v['<unk>'])
    print("en_vocab created")
    print("en_v",len(e_v))
    info_dict.update(
        create_src_token_kind_num=len(j_v), # spec 含む
        cover_src_token_num=sum(j_counter.total()),
        create_tgt_token_kind_num=len(e_v),# spec 含む
        cover_tgt_token_num=sum(e_counter.total())
    )


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
    print(info_dict)
    with open(f"{prefix_path}/info.json","w") as f:
        json.dump(info_dict,f)

    return dataset , j_v, e_v , j_tokenizer,e_tokenizer

if __name__=="__main__":
    src_max_len=128
    tgt_max_len=128
    source_vocab_max_size=8192
    target_vocab_max_size=8192
    dataset_name="jesc"

    get_dataset_and_vocab(src_max_len,tgt_max_len,source_vocab_max_size,target_vocab_max_size,dataset_name)