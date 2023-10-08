import torch.nn as nn
import torch
from torch import Tensor
from typing import Optional
from torchtext.vocab import Vocab
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.text import word_error_rate

from .base_transformer import TransformerEncoderDecoder
import torch.nn.init as init
from torchaudio.transforms import Spectrogram,TimeStretch,FrequencyMasking,TimeMasking
import random
import math
class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class Conv2dSubampling(nn.Module):
    # 参考 https://github.com/sooftware/conformer Subsampling Layaer
    def __init__(self, in_channels: int,freq_dim_magnification: int) -> None:
        super(Conv2dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, freq_dim_magnification, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(freq_dim_magnification, freq_dim_magnification, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor) -> (Tensor, Tensor):
        # input_lengths (Batch, in_channels , time , n_freq) 
        outputs:Tensor = self.sequential(inputs)
        # (Batch , freq_dim_magnification , (((time-3)//2 +1)-3)//2 +1, (((n_freq-3)//2 +1)-3)//2 +1)
        batch_size, channels, subsampled_time, subsampled_freq = outputs.size()
        # assert channels == self.freq_dim_magnification
        # assert subsampled_freq = math.floor(((math.floor((n_freq-3)/2) +1)-3)/2) +1
        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_time, channels * subsampled_freq)
        #output:(Batch, about time//4, about freq *freq_dim_magnification)
        return outputs

class DynamicTimeStretch(nn.Module):
    # min_rate ~ max_rateまで一様な確率で時間を歪める
    def __init__(self,win_hop:int,n_fft:int,min_rate:float=0.7,max_rate:float=1.3) -> None:
        super().__init__()
        assert 0<min_rate<max_rate
        self.min_rate=min_rate
        self.max_rate=max_rate
        self.stretch=TimeStretch(
                hop_length=win_hop,# Length of hop between STFT windows. (Default: win_length // 2)
                n_freq=n_fft//2 +1, # number of filter banks from stft.
            )
    def forward(self,x):
        rate=random.random()*(self.max_rate-self.min_rate) + self.min_rate
        outputs=self.stretch(x,overriding_rate=rate)
        outputs=outputs.abs().pow(2) # complex 2 power see https://pytorch.org/audio/stable/transforms.html
        return outputs

class PreLayer(nn.Module):
    #https://pytorch.org/audio/stable/transforms.html
    def __init__(self,n_fft: int = 400,win_length:int=400,win_hop:int=200,
                 encoder_dim: int = 512,input_dropout_p: float = 0.1,
                 use_spec_aug:bool=True,writer:SummaryWriter=None) -> None:
        """
            n_ftt ... n_ftt//2 +1 のbin の帯域にFFTする。
            win_lenght ... fttするときの窓の大きさ
            win_hop ... fftをするときの窓の移動の大きさ
        """
        super().__init__()
        self.writer=writer
        self.n_fft=n_fft
        self.win_length=win_length,
        self.win_hop=win_hop
        self.n_freq=n_fft//2 + 1 # number of bin
        self.use_spec_aug=use_spec_aug
        self.spectrogram= Spectrogram(
            n_fft=n_fft,
            win_length=win_length,  # (Default: n_fft)
            hop_length=win_hop, #  Length of hop between STFT windows. (Default: win_length // 2)
            power=None, #  (float or None, optional) – Exponent for the magnitude spectrogram, (must be > 0) e.g., 1 for magnitude, 2 for power, etc. If None, then the complex spectrum is returned instead. (Default: 2)
            # window_fn ... Default: torch.hann_window
        )
        print("valid などではfalseにすること")
        self.spec_aug = torch.nn.Sequential(
            DynamicTimeStretch(win_hop=win_hop,n_fft=n_fft),
            # TimeStretch(hop_length=win_hop,n_freq=n_fft//2 +1,fixed_rate=0.5),
            FrequencyMasking(
                freq_mask_param=80,#maximum possible length of the mask. Indices uniformly sampled from [0, freq_mask_param).
                iid_masks=True, # whether to apply different masks to each example/channel in the batch. (Default: False) This option is applicable only when the input tensor >= 3D.
            ), 
            TimeMasking(time_mask_param=80,iid_masks=True,p=1.0),
        )
        self.conv_subsample = Conv2dSubampling(in_channels=1, freq_dim_magnification=encoder_dim)
        self.input_projection = nn.Sequential(
            Linear(encoder_dim*(math.floor(((math.floor((self.n_freq-3)/2 +1))-3)/2 +1)), encoder_dim),
            nn.Dropout(p=input_dropout_p),
        )
        self.for_the_first_n_time=5
    def forward(self,inputs:Tensor): # TODO input_lengthの影響
        # inputs :  Tensor of audio of dimension (Batch,Channel, time). 
        # 異なるbatchに対して、time は 0 paddingで埋めておく。
        __shape_log={}
        __shape_log.update(input_shape=inputs.shape)
        outputs:Tensor=self.spectrogram(inputs)
        # outpus  : (Batch , Channel , freq, time* ), where freq is n_fft // 2 + 1 where n_fft is the number of Fourier bins, and time is the number of window hops (n_frame). dtype : complex (TimeStretchは入力がcomplexのため。)
        __shape_log.update(spectro=inputs.shape)
        if self.use_spec_aug:
            outputs=self.spec_aug(outputs)
            # (Batch, Channel , freq, time*/rate),float
        else:
            outputs=outputs.abs().pow(2)
        __shape_log.update(spec_aug=inputs.shape)
        outputs = outputs.permute(0, 1, 3, 2)
        # outputs : Batch, Channel , time*/rate , ferq) or (Batch, Channel , time, freq)
        
        outputs = self.conv_subsample(outputs)
        __shape_log.update(subsmaple=inputs.shape)
        # output: (Batch, about time*//4, about freq *freq_dim_magnification)
        outputs = self.input_projection(outputs)
        # output: (Batch ,about time*//4 , encoder_dim  )
        __shape_log.update(linearout=outputs.shape)
        if self.for_the_first_n_time>0 and self.writer!=None:
            self.writer.add_text("PreLayerShape",str(__shape_log),self.for_the_first_n_time)
        self.for_the_first_n_time-=1
        return outputs


class Audio2TokenTransformer(nn.Module):
    def __init__(self,d_model=512,src_max_len=1024,src_bin_size=40,
                  tgt_vocab_size=8192,tgt_max_len=128,pos_dropout_rate=0.1,
                  use_spec_aug:bool=False,
                  writer:SummaryWriter=None,
                  ):
        # inputデータについて、フレーム数 n , channle 1 、 次元(=フーリエ変換のbin) d (B,N,D)
        # もしくは、 生データとして、 フレーム数 n ,channel 1 (単位時間での空気の位相なので0) (B,N) を想定される。
        # FFTする場合、通常10ms ごとであり、仮に10sとすると、1000 フレームで十分である。
        # 最初のcnnの後は 40ms程度で良い。
        # フーリエ変換の便数は40程度
        # paddingの代わりに0埋めは合理的。
        super().__init__()
        # dmodel ... 埋め込みの次元数
        self.writer=writer
        self.src_max_len=src_max_len 
        self.tgt_max_len=tgt_max_len
        self.src_bin_size=src_bin_size
        self.tgt_vocab_size=tgt_vocab_size
        self.transformer=TransformerEncoderDecoder(
            d_model=d_model,
            # Conv2D input ... (B,chan,h,w) なので、
            src_pre_layer=PreLayer(use_spec_aug=use_spec_aug,writer=self.writer),
            tgt_post_class=tgt_vocab_size,
            src_max_len=src_max_len,
            tgt_max_len=tgt_max_len,
            pos_dropout_rate=pos_dropout_rate,
        )
    
    def forward(self, src_data: Tensor, tgt_token: Tensor ,
                src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,tgt_is_causal:bool=False) -> Tensor:
        # performanceのため、assertはない方が良い。
        assert (tgt_token<self.tgt_vocab_size).all()
        
        out=self.transformer(
            src_data=src_data,tgt_data=tgt_token,
            src_mask=src_mask,tgt_mask=tgt_mask,memory_mask=memory_mask,tgt_is_causal=tgt_is_causal,
        )
        return out
    
    def generate(self,src_audio:Tensor,tgt_start_token:int,tgt_end_token:int,num_beams=1):
        """
        src_dataのshapeは (L) # (N,L)の場合、最大長に達するもしくは全てが同時にeotになるまで生成し続ける
        結果について、start_tokenも含む。
        """
        if len(src_audio.shape)==2:
            src_audio=src_audio.unsqueeze(dim=0)
            print(src_audio.shape)
        result=self.transformer.generate(src_audio,tgt_start_token,tgt_end_token,num_beams)
        return result
    
    """
    ここの下は翻訳に特化した機能であり、本来は分離されるべきである。
    ただし、任意のモデルに関して
    """
    def set_train_setting(self,device,criterion,tgt_vocab:Vocab):
        self.device=device
        self.criterion=criterion
        self.tgt_vocab=tgt_vocab
        # raise NotImplementedError()
    def train_step(self,step,src_audio:Tensor,tgt_token:Tensor)->Tensor:
        # return loss
        src_audio=src_audio.to(self.device) 
        tgt_token=tgt_token.to(self.device) # (B,L)
        outputs= self(src_data=src_audio, tgt_token=tgt_token[:,:-1],tgt_is_causal=True,)
        target = nn.functional.one_hot(tgt_token[:,1:], self.tgt_vocab_size).to(torch.float32)
        # loss = criterion(outputs, target)
        loss:Tensor = self.criterion(
                    torch.reshape(outputs,(-1,self.tgt_vocab_size)),
                    torch.reshape(target,(-1,self.tgt_vocab_size))
                )
        if step!=None and step%1000==0:
            print(f"out_log[step:{step}]")
            train_print(self.writer,step,self,src_audio,tgt_token,outputs,self.tgt_vocab)
        return loss

    def valid_start(self,epoch,step):
        self.__val_gen_texts=[] # スペース区切りの予測結果
        self.__val_gold_texts=[]
        self.__val_epoch=epoch
        self.__val_step=step
    def valid_step(self,calc_loss:bool,src_audio:Tensor,tgt_token:Tensor):
        src_audio=src_audio.to(self.device) # (B,L)
        tgt_token=tgt_token.to(self.device) # (B,L)

        if calc_loss:
            loss=self.train_step(None,src_audio,tgt_token)
        else:
            loss=None

        gen_token=self.generate(src_audio,self.tgt_vocab['<bos>'],self.tgt_vocab['<eos>']).tolist()
        gen_wakati=[filter(lambda x: (x!="<pad>" and x!="<bos>" and x!="<eos>"), self.tgt_vocab.lookup_tokens(sent)) for sent in gen_token]
        gen_texts=[" ".join(sent) for sent in gen_wakati]
        self.__val_gen_texts.extend(gen_texts)

        gold_token=tgt_token.tolist()
        gold_wakati=[filter(lambda x: (x!="<pad>" and x!="<bos>" and x!="<eos>"), self.tgt_vocab.lookup_tokens(sent)) for sent in gold_token]
        gold_texts=[" ".join(sent) for sent in gold_wakati]
        self.__val_gold_texts.extend(gold_texts)
        
        # goldでは元の文を使うべきかunkこみにすべきか
        # 言語処理の観点からは元の文を使うべきだが、同じ条件下でモデル性能を比較したいなら、unkコミでも大丈夫なはず。ただし、<bos><eos><pad>はそれなりに数が多くて何も考えていなくても成果になってしまう可能性があるので、抜いておく。
        # なお、<pad>などはのぞいていることに注意する。
        return loss
    def valid_end(self):
        score=word_error_rate(self.__val_gen_texts,self.__val_gold_texts)
        self.writer.add_scalar("WER/valid",score,self.__val_step)
        print(score,self.__val_gen_texts[0],self.__val_gold_texts[0])


def train_print(writer:SummaryWriter,step:int,model:Audio2TokenTransformer,
               src_data:Tensor,tgt_token:Tensor,outputs:Tensor,
               en_v:Vocab
               ):
    # TODO srcの音声ファイル名を出力しても良さそう。
    pred_text=" ".join(filter(lambda x: x!="<pad>" ,en_v.lookup_tokens(outputs[-1,:].argmax(-1).tolist()))) 
    writer.add_text("pred",pred_text,step)
    print("pred :",pred_text)

    gen_text=" ".join(filter(lambda x: x!="<pad>" ,en_v.lookup_tokens(model.generate(src_data[-1,:],en_v['<bos>'],en_v['<eos>'])[0].tolist()))) 
    writer.add_text("generate",gen_text,step)
    print("gen :",gen_text)

    gold_text=" ".join(filter(lambda x: x!="<pad>" ,en_v.lookup_tokens(tgt_token[-1,1:].tolist()))) 
    writer.add_text("gold",gold_text,step)
    print("gold :" , gold_text)

def simpletest():
    import torch
    src_len=1024
    batch=16
    bin=80
    model=Audio2TokenTransformer(src_max_len=src_len,src_bin_size=bin).cuda()
    src=torch.rand(size=(batch,src_len,bin)).cuda()
    tgt=torch.randint(low=0, high=4096, size=(batch,60)).cuda()
    print(model)
    out:Tensor=model(src,tgt)
    print(out.shape)

    print(model.generate(src[0],0,1).shape) # 単体
    print(model.generate(src,0,1).shape) # batch

def prelayertest():
    import torchaudio
    model=PreLayer(use_spec_aug=True)
    waveform,samplingrate=torchaudio.load("/home/pika/workspace/dataset/VCC2020/VCC2020-database/source/SEF1/E10001.wav")
    waveform=waveform.unsqueeze(0)
    out=model(waveform)
    print(out.shape)

if __name__=="__main__":
    # simpletest()
    prelayertest()

    
