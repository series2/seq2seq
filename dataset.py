from torch.utils.data import Dataset
import torchtext.transforms as T
from torchtext.vocab import Vocab
from collections.abc import Callable
from torch import Tensor
class MyDataset(Dataset):
    def __call__(self,**args):
        raise NotImplementedError()
    
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
     


