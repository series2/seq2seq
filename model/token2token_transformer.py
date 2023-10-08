import torch.nn as nn
import torch
from torch import Tensor
from typing import Optional
from torchtext.vocab import Vocab
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.text import bleu_score

from base_transformer import TransformerEncoderDecoder

# やりたいこと
# Encoder Decoderを別々で学習すること
# Translate 及び 音声でモデルを作ること
# lossの計算について、新しく生成されたところのみにするかどうか
# TODO paddingの部分のマスク
# TODO bertのtokenと同時に入れるidとはなんぞや


class Token2TokenTransformer(nn.Module):
    def __init__(self,d_model=512,src_vocab_size=8192,tgt_vocab_size=8192,src_max_len=128,tgt_max_len=128,pos_dropout_rate=0.1):
        super().__init__()
        # dmodel ... 埋め込みの次元数
        self.src_max_len=src_max_len 
        self.tgt_max_len=tgt_max_len
        self.src_vocab_size=src_vocab_size
        self.tgt_vocab_size=tgt_vocab_size
        self.transformer=TransformerEncoderDecoder(
            d_model=d_model,
            src_pre_layer=nn.Embedding(src_vocab_size, d_model),
            tgt_post_class=tgt_vocab_size,
            src_max_len=src_max_len,
            tgt_max_len=tgt_max_len,
            pos_dropout_rate=pos_dropout_rate,
        )
    
    def forward(self, src_token: Tensor, tgt_token: Tensor ,
                src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,tgt_is_causal:bool=False) -> Tensor:
        # performanceのため、assertはない方が良い。
        assert (src_token<self.src_vocab_size).all() and (tgt_token<self.tgt_vocab_size).all()
        
        out=self.transformer(
            src_data=src_token,tgt_data=tgt_token,
            src_mask=src_mask,tgt_mask=tgt_mask,memory_mask=memory_mask,tgt_is_causal=tgt_is_causal,
        )
        return out
    
    def generate(self,src_token:Tensor,tgt_start_token:int,tgt_end_token:int,num_beams=1):
        """
        src_tokenのshapeは (L) # (N,L)の場合、最大長に達するもしくは全てが同時にeotになるまで生成し続ける
        結果について、start_tokenも含む。
        """
        if len(src_token.shape)==1:
            src_token=src_token.reshape((1,-1))
        result=self.transformer.generate(src_token,tgt_start_token,tgt_end_token,num_beams)
        return result
    
    """
    ここの下は翻訳に特化した機能であり、本来は分離されるべきである。
    ただし、任意のモデルに関して
    """
    def set_train_setting(self,device,criterion,src_vocab:Vocab,tgt_vocab:Vocab,writer:SummaryWriter):
        self.device=device
        self.criterion=criterion
        self.src_vocab=src_vocab
        self.tgt_vocab=tgt_vocab
        self.writer=writer
        raise NotImplementedError()
    def train_step(self,step,src_token:Tensor,tgt_token:Tensor)->Tensor:
        # return loss
        src_token=src_token.to(self.device) # (B,L)
        tgt_token=tgt_token.to(self.device) # (B,L)
        outputs= self(src_token=src_token, tgt_token=tgt_token[:,:-1],tgt_is_causal=True,)
        target = nn.functional.one_hot(tgt_token[:,1:], self.tgt_vocab_size).to(torch.float32)
        # loss = criterion(outputs, target)
        loss:Tensor = self.criterion(
                    torch.reshape(outputs,(-1,self.tgt_vocab_size)),
                    torch.reshape(target,(-1,self.tgt_vocab_size))
                )
        if step%100==0:
            print(f"out_log[step:{step}]")
            train_print(self.writer,step,self,src_token,tgt_token,outputs,self.src_vocab,self.tgt_vocab)
        return loss

    def valid_start(self,epoch,step):
        self.__val_gen_texts=[] # スペース区切りの予測結果
        self.__val_gold_texts=[]
        self.__val_epoch=epoch
        self.__val_step=step
    def valid_step(self,calc_loss:bool,src_token:Tensor,tgt_token:Tensor):
        src_token=src_token.to(self.device) # (B,L)
        tgt_token=tgt_token.to(self.device) # (B,L)

        if calc_loss:
            loss=self.train_step(src_token,tgt_token)
        else:
            loss=None

        gen_token=self.generate(src_token,self.tgt_vocab['<bos>'],self.tgt_vocab['<eos>']).tolist()
        gen_wakati=[filter(lambda x: (x!="<pad>" and x!="<bos>" and x!="<eos>"), self.tgt_vocab.lookup_tokens(sent)) for sent in gen_token]
        gen_texts=[" ".join(sent) for sent in gen_wakati]
        self.__val_gen_texts.extend(gen_texts)

        gold_token=tgt_token.tolist()
        gold_wakati=[filter(lambda x: (x!="<pad>" and x!="<bos>" and x!="<eos>"), self.tgt_vocab.lookup_tokens(sent)) for sent in gold_token]
        gold_texts=[[" ".join(sent)] for sent in gold_wakati]
        self.__val_gold_texts.extend(gold_texts)
        
        # goldでは元の文を使うべきかunkこみにすべきか
        # 言語処理の観点からは元の文を使うべきだが、同じ条件下でモデル性能を比較したいなら、unkコミでも大丈夫なはず。ただし、<bos><eos><pad>はそれなりに数が多くて何も考えていなくても成果になってしまう可能性があるので、抜いておく。
        # なお、<pad>などはのぞいていることに注意する。
        return loss
    def valid_end(self):
        score=bleu_score(self.__val_gen_texts,self.__val_gold_texts)
        self.writer.add_scalar("BLEU/valid",score,self.__val_step)
        print(score,self.__val_gen_texts[0],self.__val_gold_texts[0])


def train_print(writer:SummaryWriter,step:int,model:Token2TokenTransformer,
               src_token:Tensor,tgt_token:Tensor,outputs:Tensor,
               ja_v:Vocab,en_v:Vocab
               ):
    src_text=" ".join(filter(lambda x: x!="<pad>" ,ja_v.lookup_tokens(src_token[-1,:].tolist())))
    writer.add_text("src",src_text,step)
    print("src :",src_text)

    pred_text=" ".join(filter(lambda x: x!="<pad>" ,en_v.lookup_tokens(outputs[-1,:].argmax(-1).tolist()))) 
    writer.add_text("pred",pred_text,step)
    print("pred :",pred_text)

    gen_text=" ".join(filter(lambda x: x!="<pad>" ,en_v.lookup_tokens(model.generate(src_token[-1,:],en_v['<bos>'],en_v['<eos>'])[0].tolist()))) 
    writer.add_text("generate",gen_text,step)
    print("gen :",gen_text)

    gold_text=" ".join(filter(lambda x: x!="<pad>" ,en_v.lookup_tokens(tgt_token[-1,1:].tolist()))) 
    writer.add_text("gold",gold_text,step)
    print("gold :" , gold_text)

def simpletest():
    import torch
    model=Token2TokenTransformer().cuda()
    src=torch.randint(low=0, high=4096, size=(32,50)).cuda()
    tgt=torch.randint(low=0, high=4096, size=(32,60)).cuda()
    print(model)
    out:Tensor=model(src,tgt)
    print(out.shape)

    print(model.generate(src[0],0,1).shape) # 単体
    print(model.generate(src,0,1).shape) # batch

if __name__=="__main__":
    simpletest()

    
