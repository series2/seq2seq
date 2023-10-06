import torch.nn as nn
import torch
from torch import Tensor
from positional_embedding import PositionalEncoding
import math
from typing import Optional

# やりたいこと
# Encoder Decoderを別々で学習すること
# Translate 及び 音声でモデルを作ること
# lossの計算について、新しく生成されたところのみにするかどうか
# TODO paddingの部分のマスク
# TODO bertのtokenと同時に入れるidとはなんぞや

class Token2TokenTransformer(nn.Module):
    def __init__(self,d_model=512,src_vocab_size=1024,tgt_vocab_size=1024,src_max_len=512,tgt_max_len=512,pos_dropout_rate=0.1):
        super().__init__()
        # dmodel ... 埋め込みの次元数
        self.d_model=d_model
        self.src_max_len=src_max_len # including special token
        self.tgt_max_len=tgt_max_len # including special token
        self.enc_embed = nn.Embedding(src_vocab_size, d_model)
        self.enc_pos_encoder= PositionalEncoding(d_model, pos_dropout_rate,max_len=src_max_len)
        # TODO 位置埋め込みの最大長とshape,dropoutのrateの一般的な値の調査 
        # 0.5になるとデカすぎるので、例えば、[1,2,3]みたいなクソ簡単なデータしかない時でも、位置がわからず全部同じ値になる傾向がある。
        # また、今回は位置に対して固定値だが、あまりに全体の系列長が小さいとその変化が少なく無視されてしまうので、max_lenは適切に調整する必要がある。また、max_lenは位置に対して長くなると情報量が相対的に減ってしまうので、次元を増やすとか工夫が必要だし、そもそも長くなると最初の方の情報は関係なくなるので何かしら良い方法を探したほうが良い。
        self.dec_embed = nn.Embedding(tgt_vocab_size, d_model)

        self.dec_pos_encoder= PositionalEncoding(d_model, pos_dropout_rate,max_len=tgt_max_len)
        
        self.transformer = nn.Transformer(d_model=d_model,batch_first=True)
        self.dec_classifier = nn.Linear(d_model,tgt_vocab_size)
    
    def forward(self, src_token: Tensor, tgt_token: Tensor ,
                src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,tgt_is_causal:bool=False) -> Tensor:
        src=self.enc_embed(src_token)* math.sqrt(self.d_model)
        src=self.enc_pos_encoder(src)
        tgt=self.dec_embed(tgt_token)* math.sqrt(self.d_model)
        tgt=self.dec_pos_encoder(tgt)
        
        if tgt_is_causal and tgt_mask!=None:
            raise Exception("どちらか一方のみ使用可能。")
        if tgt_is_causal:
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(self.tgt_max_len).cuda() # TODO cuda?
            # tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf],
            #     [0., 0., -inf, -inf, -inf, -inf, -inf],
            #     [0., 0., 0., -inf, -inf, -inf, -inf],
            #     [0., 0., 0., 0., -inf, -inf, -inf],
            #     [0., 0., 0., 0., 0., -inf, -inf],
            #     [0., 0., 0., 0., 0., 0., -inf],
            #     [0., 0., 0., 0., 0., 0., 0.]])
            
        out=self.transformer(src=src, tgt=tgt,src_mask=src_mask,tgt_mask=tgt_mask,memory_mask=memory_mask)
        out=self.dec_classifier(out)

        return out
    
    def generate(self,src_token:Tensor,start_token:int,end_token:int,num_beams=1):
        """
        src_tokenのshapeは (L) # (N,L)の場合、最大長に達するもしくは全てが同時にeotになるまで生成し続ける
        結果について、start_tokenも含む。
        """
        if num_beams>1:
            raise NotImplementedError()
        if len(src_token.shape)==1:
            src_token=src_token.reshape((1,-1))
        batch=src_token.shape[0]
        result=torch.ones((len(src_token),1),dtype=torch.int64)*start_token
        result=result.cuda() # TODO device?
        src_token=src_token.cuda() # TODO device?
        for _ in range(self.tgt_max_len):
            src=self.enc_embed(src_token)* math.sqrt(self.d_model)
            src=self.enc_pos_encoder(src)
            tgt=self.dec_embed(result)* math.sqrt(self.d_model)
            tgt=self.dec_pos_encoder(tgt)
                
            out=self.transformer(src=src, tgt=tgt)
            out=self.dec_classifier(out)
            append_result=out[:,-1:].argmax(-1)
            result=torch.concat([result,append_result],axis=1)
            if torch.isin(append_result, torch.tensor([end_token]).cuda()).all():
                return result
        return result

def simpletest():
    import torch
    model=Token2TokenTransformer().cuda()
    src=torch.randint(low=0, high=4096, size=(32,130)).cuda()
    tgt=torch.randint(low=0, high=4096, size=(32,130)).cuda()
    print(model)
    out=model(src,tgt)
    print(out.shape)

def test():
    import torch.optim as optim
    num=10
    # src=torch.arange(10*30).reshape(10,30).cuda()    
    src=torch.zeros((num,3),dtype=torch.int64).cuda()+1
    tgt_max_len=30
    # tgt=1000-torch.arange(num*tgt_max_len).reshape(num,tgt_max_len).cuda()
    tgt=torch.tensor([1, 2, 3]).repeat(num, 10).cuda()
    # tgt=torch.zeros((num,tgt_max_len),dtype=torch.int64).cuda()+1
    # tgt=1000-torch.arange(10*40).reshape(10,40).cuda()
    print(src)
    print(tgt,tgt.dtype)
    # exit()
    params=dict(
        d_model=512,
        src_vocab_size=1024,
        tgt_vocab_size=10,
        src_max_len=512,
        tgt_max_len=tgt_max_len,
    )

    model=Token2TokenTransformer(**params).cuda()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    print(model)
    model.train()
    for i in range(100):
        out=model(src,tgt,tgt_is_causal=False)
        target = nn.functional.one_hot(tgt,params["tgt_vocab_size"]).to(torch.float32)

        # loss = criterion(out, target) 
        # これはダメっぽい。相対値はあっているが、絶対値が違う感じがする 。他の人は1つ出すごとに計算していた。
        loss = criterion(
            torch.reshape(out.softmax(dim=-1),(-1,params["tgt_vocab_size"])),
            torch.reshape(target,(-1,params["tgt_vocab_size"]))
        )
        print(loss.item())
        print(out.argmax(-1))
        print()
        loss.backward()
        optimizer.step()
    
    print(target[0])
    print(out[0])

def test2(): # generateの動作テスト
    import torch.optim as optim
    from timm.scheduler import CosineLRScheduler
    num=100
    src_vocab_size=100
    vocab_size=100

    src=torch.concat([torch.randint(3,100,(num,1)).repeat(1,3) ,torch.randint(3,100,(num,1)).repeat(1,3) ],axis=1)
    tgt_max_len=8
    tgt=torch.concat([
                torch.zeros((num,1),dtype=torch.int64),
                # src,((src+1)%(num-2))+2,
                ((src+1)%(vocab_size-2))+2,
                torch.ones((num,1),dtype=torch.int64),
                ],axis=1)
    tgt_max_len=tgt.shape[1]
    src_max_len=src.shape[1]
    src=src.cuda()
    tgt=tgt.cuda()
    print("src;",src)
    print("tgt",tgt,tgt.dtype)
    params=dict(
        d_model=512,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=vocab_size,
        src_max_len=src_max_len,
        tgt_max_len=tgt_max_len-1,
    )
    epoch=1000
    model=Token2TokenTransformer(**params).cuda()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=5e-6)
    scheduler = CosineLRScheduler(optimizer, t_initial=epoch, lr_min=1e-7,
                              warmup_t=1, warmup_lr_init=1e-7, warmup_prefix=True) 
    print(model)
    model.train()
    for i in range(epoch):
        print(src.shape,tgt.shape)
        out=model(src,tgt[:,:-1],tgt_is_causal=False)
        target = nn.functional.one_hot(tgt,params["tgt_vocab_size"]).to(torch.float32)
        loss = criterion(
            torch.reshape(out,(-1,params["tgt_vocab_size"])),
            torch.reshape(target[:,1:],(-1,params["tgt_vocab_size"]))
        )
        print(loss.item())
        print("pred :",out.argmax(-1))
        print("gold :",tgt)

        loss.backward()
        optimizer.step()
        scheduler.step(i)

        print(optimizer.param_groups[0]['lr'])
    print(target[0])
    print(out[0])

    print("end train")
    for i in range(10):
        print(i)
        print("input:",src[i])
        print("gold:",tgt[i,1:])
        print("pred:",model.generate(src[i],0,1))


if __name__=="__main__":
    test()
    test2()