import matplotlib.pyplot as plt
import pandas as pd
from torchtext.vocab import Vocab
from janome.tokenizer import Tokenizer
import spacy
from collections.abc import Callable
import os
import pickle
import json

from dataset import ASRDataset
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

def show_len_dist(len_dist:list[int],prefix_path:str,filename:str):
    #  TODO 何かがおかしいのとみにくい
    # 文書ごとのトークン長の分布
    # 単語種ごとの数の分布もみたい
    plt.hist(len_dist)
    # plt.xscale("log");plt.yscale("log")
    plt.savefig(f"{prefix_path}/{filename}")

# 現状はtokenizerなどは固定である。
def get_dataset_and_vocab(tgt_max_len,target_vocab_max_size,dataset_name)->tuple[ASRDataset,ASRDataset|None,ASRDataset|None,Vocab,Callable[[str],list[str]]]:
    if not os.path.exists("preprocessed_data"):
        os.mkdir("preprocessed_data")
    prefix_path=f"preprocessed_data/{dataset_name}"
    if not os.path.exists(prefix_path):
        os.mkdir(prefix_path)
    info_dict={}
    #日本語用のトークン変換関数を作成
    # global j_t # 並列処理のため
    global e_t # 並列処理のため 5枚以上程度早くなる気がする。
    # j_t = Tokenizer()
    # def j_tokenizer(text:str): 
    #     return [tok for tok in j_t.tokenize(text, wakati=True)]

    #英語用のトークン変換関数を作成
    e_t = spacy.load('en_core_web_sm')
    def e_tokenizer(text:str):
        return [tok.text for tok in e_t.tokenizer(text)]

    print(f"data load start. name: `{dataset_name}`")
    train_audio_paths=None
    train_audio_texts=None
    valid_audio_paths=None
    valid_audio_texts=None
    test_audio_paths=None
    test_audio_texts=None
    if False:
        pass
    elif dataset_name=="librispeech-100":
        # 階層 train-clean-100/19/277/19-277-0000.flac .... 19-227.trans.txt(19-227-0000[space]Text(Space区切り))
        train_audio_paths=[]
        train_audio_texts=[]
        valid_audio_paths=[]
        valid_audio_texts=[]
        from torchaudio.datasets import LIBRISPEECH
        # textsについてtokenizationを別で行うため
        # このデータクラスは使用しない。
        pre="/mnt/hdd/dataset/LibriSpeech/LibriSpeech"
        __train_dataset=LIBRISPEECH(root="/mnt/hdd/dataset/LibriSpeech",url="train-clean-100",download=False)
        for i in range(len(__train_dataset)):
            path,_,trans,_,_,_=__train_dataset.get_metadata(i)
            train_audio_paths.append(os.path.join(pre,path))
            train_audio_texts.append(trans)
        __val_dataset=LIBRISPEECH(root="/mnt/hdd/dataset/LibriSpeech",url="dev-clean",download=False)
        for i in range(len(__val_dataset)):
            path,_,trans,_,_,_=__val_dataset.get_metadata(i)
            valid_audio_paths.append(os.path.join(pre,path))
            valid_audio_texts.append(trans)
    else:
        raise Exception(f"dataset `{dataset_name}` not exists")
    print("data loaded")
    data=list(zip(train_audio_paths,train_audio_texts))
    df=pd.DataFrame(data,columns=["src_token","tgt_token"])

    tgt_vocab_pikle=f"{prefix_path}/tgt_vocab.pickle"
    if os.path.isfile(tgt_vocab_pikle):
        with open(tgt_vocab_pikle, 'rb') as f:
            e_v,tgt_info,tgt_token_len_dist = pickle.load(f)
    else:
        e_v,tgt_info,tgt_token_len_dist=create_vocab(df["tgt_token"],e_tokenizer,target_vocab_max_size)
        with open(tgt_vocab_pikle, 'wb') as f:
            pickle.dump((e_v,tgt_info,tgt_token_len_dist), f)
    info_dict.update(
        {f"tgt_{key}":value for key,value in  tgt_info.items()}
    )
    show_len_dist(tgt_token_len_dist,prefix_path,"tgt_token_dist.png")
    print("tokenize end")

    train_dataset= ASRDataset(train_audio_paths,train_audio_texts,e_v,e_tokenizer,tgt_max_len)
    valid_dataset= ASRDataset(valid_audio_paths,valid_audio_texts,e_v,e_tokenizer,tgt_max_len)if valid_audio_paths!=None and valid_audio_texts!=None else None
    test_dataset= ASRDataset(test_audio_paths,test_audio_texts,e_v,e_tokenizer,tgt_max_len) if test_audio_paths!=None and test_audio_texts!=None else None
    print("preprocess ended")
    info_dict.update(
        have_validdata=valid_dataset!=None,
        have_testdata=test_dataset!=None,
        )
    print(info_dict)
    info_path=f"{prefix_path}/info.json"
    if not os.path.exists(info_path):
        with open(info_path,"w") as f:
            print(f"save to {info_path}")
            json.dump(info_dict,f)

    return train_dataset ,valid_dataset,test_dataset, e_v , e_tokenizer

if __name__=="__main__":
    tgt_max_len=128
    source_vocab_max_size=32768
    target_vocab_max_size=32768
    dataset_name="jesc"

    get_dataset_and_vocab(tgt_max_len,source_vocab_max_size,target_vocab_max_size,dataset_name)