import matplotlib.pyplot as plt
import pandas as pd
from torchtext.vocab import Vocab
from janome.tokenizer import Tokenizer
import spacy
from collections.abc import Callable
import os
import pickle
import json

from dataset import NormalMT
from vocab import create_vocab

from pandarallel import pandarallel
pandarallel.initialize()

# Dataの流れ
# Translationの場合
# train,valid,testはそれぞれ src,targetの組みが1問として存在
# src,tgtのtrainを元に src,tgt用のtokenizerがそれぞれ作成される。ただし、すでにある何らかのtokenizerを使用することも可能である。
# tokenizerを使用して train_dataset,valid,testが作成される。
# それぞれ、src,tgtのtoken配列を出力する。
# datasetに関して、1度にメモリに乗せるのが辛い時は、preprocessed_data/ にストレージとして保存する。候補1 ... swap メモリ 2 ... 1問ずつファイルへ 3 ... sqlite
# 特に 2, 3について後日速度比較を行う(テキスト、動画、画像、音声について)
# と思ったが、3についてBLOB型はencode,decodeが必要であり、遅そうである。
# Speech 2 Textの場合
# train,valid,testはそれぞれ src_audio,target_tokenの組みが1問として存在
# target_trainを元に tokenizerが作成される。ただし、すでにある何らかのtokenizerを使用することも可能である。
# また、音声に関しては、多分重そうなので、とりあえず読み取るたびにwav から tensor型に変換する。
# tokenizerを使用して train_dataset,valid,testが作成される。
# それぞれ、src,tgtのtoken配列を出力する。
# また、入力と同じフォーマットの情報(単体、複数Iに関しては__call__ を提供すること。

# ここではtrain,valid,testで一組みのデータセットとする。
# そのため、もしいずれかひとつ変更したい場合、新しいデータセットを定義する必要がある。
# valid,testに対して複数のデータベースをことなるメトリクスのために定義することはできない。

# valid,testに関して、統計データを出したいが、後回し

def show_len_dist(src_texts:list[str],tgt_texts:list[str],prefix_path:str):
    #  TODO 何かがおかしいのとみにくい
    # 文書ごとのトークン長の分布
    # 単語種ごとの数の分布もみたい
    plt.hist(list(map(len,src_texts)))
    plt.xscale("log");plt.yscale("log")
    plt.savefig(f"{prefix_path}/src_token_dist.png")
    plt.hist(list(map(len,tgt_texts)))
    plt.xscale("log");plt.yscale("log")
    plt.savefig(f"{prefix_path}/tgt_token_dist.png")

# 現状はtokenizerなどは固定である。
def get_dataset_and_vocab(src_max_len,tgt_max_len,source_vocab_max_size,target_vocab_max_size,dataset_name)->tuple[NormalMT,NormalMT|None,NormalMT|None,Vocab,Vocab,Callable[[str],list[str]],Callable[[str],list[str]]]:
    if not os.path.exists("preprocessed_data"):
        os.mkdir("preprocessed_data")
    prefix_path=f"preprocessed_data/{dataset_name}"
    if not os.path.exists(prefix_path):
        os.mkdir(prefix_path)
    info_dict={}
    #日本語用のトークン変換関数を作成
    global j_t # 並列処理のため
    global e_t # 並列処理のため 5枚以上程度早くなる気がする。
    j_t = Tokenizer()
    def j_tokenizer(text:str): 
        return [tok for tok in j_t.tokenize(text, wakati=True)]

    #英語用のトークン変換関数を作成
    e_t = spacy.load('en_core_web_sm')
    def e_tokenizer(text:str):
        return [tok.text for tok in e_t.tokenizer(text)]

    print(f"data load start. name: `{dataset_name}`")
    ja_train_texts=None
    en_train_texts=None
    ja_valid_texts=None
    en_valid_texts=None
    ja_test_texts=None
    en_test_texts=None
    if False:
        pass
    elif dataset_name=="kftt" or dataset_name=="kftt_16k" or dataset_name=="kftt_32k":
        pref="/mnt/hdd/dataset/kftt-data-1.0/data/orig"
        with open(os.path.join(pref,"kyoto-train.ja"),"r") as f:
            ja_train_texts=f.readlines()
        ja_train_texts=list(map(lambda x:x.strip(),ja_train_texts))
        with open(os.path.join(pref,"kyoto-train.en"),"r") as f:
            en_train_texts=f.readlines()
        en_train_texts=list(map(lambda x:x.strip(),en_train_texts))
        with open(os.path.join(pref,"kyoto-dev.ja"),"r") as f:
            ja_valid_texts=f.readlines()
        ja_valid_texts=list(map(lambda x:x.strip(),ja_valid_texts))
        with open(os.path.join(pref,"kyoto-dev.en"),"r") as f:
            en_valid_texts=f.readlines()
        en_valid_texts=list(map(lambda x:x.strip(),en_valid_texts))
        with open(os.path.join(pref,"kyoto-test.ja"),"r") as f:
            ja_test_texts=f.readlines()
        ja_test_texts=list(map(lambda x:x.strip(),ja_test_texts))
        with open(os.path.join(pref,"kyoto-test.en"),"r") as f:
            en_test_texts=f.readlines()
        en_test_texts=list(map(lambda x:x.strip(),en_test_texts))
    elif dataset_name=="JEC":
        df = pd.read_excel("./JEC_basic_sentence_v1-3.xls", header = None)
        ja_train_texts=df.iloc[:,1].to_list()
        en_train_texts=df.iloc[:,2].to_list()
    elif dataset_name=="jesc":
        df=pd.read_table("/mnt/hdd/dataset/jesc/split/train",header=None,names=["en","ja"])
        ja_train_texts=df["ja"].to_list()
        en_train_texts=df["en"].to_list()
        df=pd.read_table("/mnt/hdd/dataset/jesc/split/dev",header=None,names=["en","ja"])
        ja_valid_texts=df["ja"].to_list()
        en_valid_texts=df["en"].to_list()
        df=pd.read_table("/mnt/hdd/dataset/jesc/split/test",header=None,names=["en","ja"])
        ja_test_texts=df["ja"].to_list()
        en_test_texts=df["en"].to_list()
    else:
        raise Exception(f"dataset `{dataset_name}` not exists")
    show_len_dist(ja_train_texts,en_train_texts,prefix_path)
    print("data loaded")
    data=list(zip(ja_train_texts,en_train_texts))
    df=pd.DataFrame(data,columns=["src_token","tgt_token"])

    src_vocab_pikle=f"{prefix_path}/src_vocab.pickle"
    tgt_vocab_pikle=f"{prefix_path}/tgt_vocab.pickle"
    if os.path.isfile(src_vocab_pikle):
        with open(src_vocab_pikle, 'rb') as f:
            j_v,src_info= pickle.load(f)
    else:
        j_v,src_info=create_vocab(df["src_token"],j_tokenizer,source_vocab_max_size)
        with open(src_vocab_pikle, 'wb') as f:
            pickle.dump((j_v,src_info), f)
    info_dict.update(
        {f"src_{key}":value for key,value in  src_info.items()}
    )
    if os.path.isfile(tgt_vocab_pikle):
        with open(tgt_vocab_pikle, 'rb') as f:
            e_v,tgt_info = pickle.load(f)
    else:
        e_v,tgt_info=create_vocab(df["tgt_token"],e_tokenizer,target_vocab_max_size)
        with open(tgt_vocab_pikle, 'wb') as f:
            pickle.dump((e_v,tgt_info), f)
    info_dict.update(
        {f"tgt_{key}":value for key,value in  tgt_info.items()}
    )

    print("tokenize end")

    train_dataset = NormalMT(ja_train_texts,en_train_texts,j_v,e_v,j_tokenizer,e_tokenizer,src_max_len,tgt_max_len)
    valid_dataset=NormalMT(ja_valid_texts,en_valid_texts,j_v,e_v,j_tokenizer,e_tokenizer,src_max_len,tgt_max_len) if ja_valid_texts!=None and en_valid_texts!=None else None
    test_dataset=NormalMT(ja_test_texts,en_test_texts,j_v,e_v,j_tokenizer,e_tokenizer,src_max_len,tgt_max_len) if ja_test_texts!=None and en_test_texts!=None else None
    print("preprocess ended")
    info_dict.update(
        have_validdata=valid_dataset!=None,
        have_testdata=test_dataset!=None,
        )
    print(info_dict)
    with open(f"{prefix_path}/info.json","w") as f:
        print(f"save to {prefix_path}/info.json")
        json.dump(info_dict,f)

    return train_dataset ,valid_dataset,test_dataset, j_v, e_v , j_tokenizer,e_tokenizer

if __name__=="__main__":
    
    # src_max_len=128
    # tgt_max_len=128
    # source_vocab_max_size=8192
    # target_vocab_max_size=8192
    # dataset_name="jesc"

    # src_max_len=128
    # tgt_max_len=128
    # source_vocab_max_size=8192
    # target_vocab_max_size=8192
    # dataset_name="kftt"

    # src_max_len=128
    # tgt_max_len=128
    # source_vocab_max_size=16384
    # target_vocab_max_size=16384
    # dataset_name="kftt_16k"

    # src_max_len=128
    # tgt_max_len=128
    # source_vocab_max_size=32768
    # target_vocab_max_size=32768
    # dataset_name="kftt_32k"

    src_max_len=128
    tgt_max_len=128
    source_vocab_max_size=32768
    target_vocab_max_size=32768
    dataset_name="jesc"

    get_dataset_and_vocab(src_max_len,tgt_max_len,source_vocab_max_size,target_vocab_max_size,dataset_name)