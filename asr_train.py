import torch.nn as nn
import torch.optim as optim
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler
from torch.nn.utils.rnn import pad_sequence
import os
from torchtext.vocab import Vocab

# from model.token2token_transformer import Token2TokenTransformer
from model.audio2token_transformer import Audio2TokenTransformer
from asr_preprocess import get_dataset_and_vocab
import tqdm

def collate_fn(data:list[(Tensor,Tensor)])->(Tensor,Tensor):
    # 注意　こうするとバッチごとにメモリ使用率が変わるからあんまりやりたくない。本当はmaxの長さを把握しておくと良い。
    audios=[]
    audio_chan=len(data[0]["src_audio"])
    batch_len=len(data)
    scripts=[]
    for d in data:
        for a in d["src_audio"]:
            audios.append(a)
        scripts.append(d["tgt_token"])
    audio=pad_sequence(audios,batch_first=True)
    audio=audio.reshape(batch_len,audio_chan,-1)
    script=torch.stack(scripts)
    return {"src_audio":audio,"tgt_token":script}

def train(
        dataset_name="JEC",sufix_exp_folder="exp2_epoch500",
        epoch_num = 500,# 10
        
        # データセット、問題設定
        source_max_length=128, # <eos><bos>含まず
        target_max_length=128, # <eos><bos>含まず
        # source_vocab_max_size=8192,# special token 含む
        target_vocab_max_size=8192,# special token 含む
        lr=1e-5,lr_min=1e-7,# lr_minはスケジューラによる最終値
        warmup_t=100, warmup_lr_init=1e-7, 
        batch_size=32,
        get_val_loss:bool=True,#validの存在するデータの場合
        ):
    exp_dir=f"runs/middle/{dataset_name}/{sufix_exp_folder}"
    if os.path.exists(exp_dir):
        l=input(f"folder {exp_dir} exists. Do you want to overwrite it? (y/n):")
        if l=="y" or l=="yes" or l =="Y" :
            print("overwrite.")
        else:
            raise Exception("上書きしないので終了します。")
    print("preprocess start")
    writer = SummaryWriter(exp_dir)
    train_dataset,valid_dataset,test_dataset,en_v,e_tok = get_dataset_and_vocab(target_max_length,target_vocab_max_size,dataset_name=dataset_name)
    target_vocab_max_size=min(target_vocab_max_size,len(en_v))
    # 学習関係
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=Audio2TokenTransformer(
        # src_bin_size=40,
        # src_max_len=128,
        tgt_vocab_size=target_vocab_max_size,
        # src_max_len=source_max_length+2, # 前後のtoken
        tgt_max_len=target_max_length+1, # 前後のtoken
        use_spec_aug=True,
        writer=writer
        ).to(device)
    print("model loaded")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
    valid_dataloader=DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_fn) if valid_dataset!=None else None
    test_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_fn) if valid_dataset!=None else None
    scheduler = CosineLRScheduler(optimizer,
                                t_initial=epoch_num*len(train_dataloader), lr_min=lr_min,
                                warmup_t=warmup_t, warmup_lr_init=warmup_lr_init, warmup_prefix=True) 


    writer.add_text("meta_data",str(dict(
                dataset_name=dataset_name,sufix_exp_folder=sufix_exp_folder,
        epoch_num = epoch_num,
        # データセット、問題設定
        src_max_length=source_max_length, # <eos><bos>含まず
        tgt_max_length=target_max_length, # <eos><bos>含まず
        # src_vocab_max_size=source_vocab_max_size,
        tgt_vocab_max_size=target_vocab_max_size,
        batch_size=batch_size,
        approxmate_totalstep=epoch_num*len(train_dataloader), 
        soruce_sentence_num=len(train_dataset),
        tgt_vocab_size=len(en_v),
        lr=lr,
    )))
    step = 0
    train_loss = 0

    print("train start")
    model.set_train_setting(device,criterion,en_v)
    for epoch in range(epoch_num):
        model.train()
        for i, data in enumerate(train_dataloader):
            step+=1
            optimizer.zero_grad()
            loss=model.train_step(step,**data)
            loss.backward()
            optimizer.step()
            scheduler.step(step)
        
            train_loss += loss.item()
            writer.add_scalar("Loss/train",loss.item(),step)
            writer.add_scalar("lr",scheduler._get_lr(step)[0],step)# 要素ひとつしかない。

            if i % 10 ==  0:
                print(f"epoch:{epoch+1}  index:{i+1}  loss:{train_loss/10:.10f}")
                train_loss = 0

            model.eval()
            with torch.no_grad():
                if valid_dataset!=None and i % 10000 ==  0:
                    model.valid_start(epoch,step)
                    for i, data in tqdm.tqdm(enumerate(valid_dataloader),total=len(valid_dataloader)): 
                        loss=model.valid_step(get_val_loss,**data)
                        if get_val_loss:
                            writer.add_scalar("Loss/valid",loss.item(),step)
                    model.valid_end()

    writer.close()
    torch.save(model.state_dict(),os.path.join(exp_dir,"audio2text_middle.pth"))

if __name__=="__main__":
    train(dataset_name="librispeech-100",
        sufix_exp_folder="exp1",
        target_max_length=64,
        # source_vocab_max_size=32768,
        target_vocab_max_size=32768,
        batch_size=4,
        lr=1e-5,lr_min=1e-7,
        warmup_lr_init=1e-7,warmup_t=1000,
        epoch_num=10)

    
