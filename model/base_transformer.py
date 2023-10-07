import torch.nn as nn
import torch
from torch import Tensor
from positional_embedding_old import PositionalEncoding
import math
from typing import Optional

class TransformerEncoderDecoder(nn.Module):
    def __init__(self,
                 d_model:int=512,
                 src_pre_layer:nn.Module=None,
                 custom_encoder:nn.Module=None,
                #  tgt_pre_layer:nn.Module=None,
                 custom_decoder:nn.Module=None,
                #  tgt_post_layer:nn.Module=None,
                 tgt_post_class:int=8192,
                 src_max_len:int=128,
                 tgt_max_len:int=128,
                 pos_dropout_rate=0.1):
        """
        d_modelについて、positional encodingの次元およびtransformer内部の次元に相当します。cusotm_encoder,custom_decoderを使用する際は整合性に気をつけてください。
        enc/tgt_pre_layerは Embedding層(NLP)やCNN(音声)を想定します。
        tgt_pre_layer は classifier層(NLP,ASR)を想定します。
        
        [TODO] なお、decoderのinput,outputはtokenのみを想定します。
        今後、音声出力などにも対応できるようにする予定です。

        [TODO] 今後の可能性
        inputについて、embeddingで入れられるようにする (enc,dec)
        outputについて、embeddingで取得できるようにする(enc,dec) (for KD)
        bertなどで用いられるinputと同時に入れるidについて考える
        """
        super().__init__()
        self.d_model=d_model
        self.src_pre_layer=src_pre_layer
        self.transformer = nn.Transformer(custom_encoder=custom_encoder,custom_decoder=custom_decoder,batch_first=True)
        # self.tgt_pre_layer=tgt_pre_layer
        self.tgt_pre_layer=nn.Embedding(tgt_post_class,d_model)
        self.src_max_len=src_max_len
        self.tgt_pos_encoder= PositionalEncoding(d_model, pos_dropout_rate,max_len=tgt_max_len)
        self.tgt_max_len=tgt_max_len
        self.src_pos_encoder= PositionalEncoding(d_model, pos_dropout_rate,max_len=src_max_len)
        # self.tgt_post_layer=tgt_post_layer
        self.tgt_post_layer=nn.Linear(d_model,tgt_post_class)
        # self.tgt_post_class=tgt_post_class for debug
        
    
    def forward(self, src_data: Tensor, tgt_data: Tensor ,
                src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,tgt_is_causal:bool=False) -> Tensor:

        if self.src_pre_layer!=None:
            src_data=self.src_pre_layer(src_data)* math.sqrt(self.d_model)
        src_data=self.src_pos_encoder(src_data)
        if self.tgt_pre_layer!=None:
            tgt_data=self.tgt_pre_layer(tgt_data)* math.sqrt(self.d_model)
        tgt_data=self.tgt_pos_encoder(tgt_data)
        
        if tgt_is_causal and tgt_mask!=None:
            raise Exception("どちらか一方のみ使用可能。")
        if tgt_is_causal:
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(self.tgt_max_len).to(src_data.device)
            
        out=self.transformer(src=src_data, tgt=tgt_data,src_mask=src_mask,tgt_mask=tgt_mask,memory_mask=memory_mask)
        if self.tgt_post_layer!=None:
            out=self.tgt_post_layer(out)
        return out
    
    def generate(self,src_data:Tensor,
                 tgt_start_token:int,tgt_end_token:int,
                #  tgt_start_data:Tensor,tgt_end_data:Tensor,
                 num_beams=1
                 )->Tensor:
        """
        src_tokenのshapeは (L) # (N,L)の場合、最大長に達するもしくは全てが同時にeotになるまで生成し続ける
        結果出力はstart_tokenも含む。
        [TODO] 現在、tgtはtokenのみしか対応していません。
        """
        self.eval()
        with torch.no_grad():
            if num_beams>1:
                raise NotImplementedError()
            if len(src_data.shape)==1:
                src_data=src_data.reshape((1,-1))
            # result=tgt_start_data
            result=torch.ones((len(src_data),1),dtype=torch.int64)*tgt_start_token
            result=result.to(src_data.device)
            for _ in range(self.tgt_max_len):
                pred_token=self(src_data,result)[:,-1:].argmax(-1)
                result=torch.concat([result,pred_token],axis=1)
                if torch.isin(pred_token, torch.tensor([tgt_end_token]).cuda()).all():
                    return result
        return result

def simpletest():
    import torch
    d_model=512
    tgt_post_class=8192
    batch=32
    model=TransformerEncoderDecoder(d_model=d_model).cuda()
    src=torch.rand(size=(batch,80,d_model)).cuda()
    tgt=torch.randint(low=0, high=tgt_post_class, size=(batch,90)).cuda()
    print(model)
    out:Tensor=model(src,tgt)
    print(out.shape)

if __name__=="__main__":
    simpletest()