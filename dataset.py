from torch.utils.data import Dataset
import torchtext.transforms as T
from torchtext.vocab import Vocab
from collections.abc import Callable
from torch import Tensor
import torchaudio
class MyDataset(Dataset):
    def __call__(self,**args):
        raise PendingDeprecationWarning("将来的に廃止見込み")
        raise NotImplementedError()
    

class ASRDataset(MyDataset):
    def __init__(self,
                 src_audio_files:list[str],tgts:list[str],
                 tgt_vocab:Vocab,
                 tgt_tokenizer:Callable[[str],list[str]],
                 tgt_max_len:int,
                 ):
        assert len(src_audio_files)==len(tgts)
        print("注意:waweファイルはモデルの構造上現在1chのみしか使えません。")
        self.src_audio_files=src_audio_files # ファイルをメモリに乗っけるのはコストがかかりそうなので毎回読み取ることにする。
        self.tgt_data=NormalText(tgts,tgt_vocab,tgt_tokenizer,tgt_max_len)
    def __getitem__(self, i) -> dict[str,str]:
        waveform, sample_rate = torchaudio.load(self.src_audio_files[i]) # waveform shape : (C,L)
        data = {"src_audio": waveform, "tgt_token": self.tgt_data[i]["ids"],
                # "tgt_wakati":self.tgt_data[i]["tokens"]
                }
        return data
    def __len__(self):
        return len(self.src_audio_files)
    
class NormalMT(MyDataset):
    def __init__(self,
                 srcs:list[str],tgts:list[str],
                 src_vocab:Vocab,tgt_vocab:Vocab,
                 src_tokenizer:Callable[[str],list[str]],tgt_tokenizer:Callable[[str],list[str]],
                 src_max_len:int,tgt_max_len:int,
                 ):
        assert len(srcs)==len(tgts)
        self.src_data=NormalText(srcs,src_vocab,src_tokenizer,src_max_len)
        self.tgt_data=NormalText(tgts,tgt_vocab,tgt_tokenizer,tgt_max_len)
    def __getitem__(self, i) -> dict[str,str]:
        data = {"src_token": self.src_data[i]["ids"], "tgt_token": self.tgt_data[i]["ids"],
                # "tgt_wakati":self.tgt_data[i]["tokens"]
                }
        return data
    def __len__(self):
        return len(self.src_data)
    def __call__(self,src:str,tgt:str):
        data = {"src_token": self.src_data(src), "tgt_token": self.tgt_data(tgt)}
        return data

class NormalText(MyDataset):
    def __init__(self,
                 texts:list[str],
                 vocab:Vocab,
                 tokenizer:Callable[[str],list[str]],
                 max_len:int):
        """
        texts : 分割されていないtextの配列
        TODO もし、tokenizerが時間がかかるならinitでしてしまう。
        """
        self.tokenizer:Callable[[str],list[str]]=tokenizer
        self.transform:Callable[[list[str]],Tensor]=T.Sequential(
            T.VocabTransform(vocab),   #トークンに変換
            T.Truncate(max_len),   #長い場合は切る    
            T.AddToken(token=vocab['<bos>'], begin=True),   #文頭に'<bos>
            T.AddToken(token=vocab['<eos>'], begin=False),   #文末に'<eos>'を追加
            T.ToTensor(), 
            T.PadTransform(max_len + 2, vocab['<pad>']))
        self.texts=texts
    def __getitem__(self, i) -> dict[str,str]:
        return {"ids":self(self.texts[i]),
                # tokens":self.texts[i]
                }
    def __len__(self):
        return len(self.texts)
    def __call__(self,text):
        tokens=self.tokenizer(text)
        return self.transform([tokens]).squeeze()
     

def test():
    audio_dataset=ASRDataset(src_audio_files=[""],)
