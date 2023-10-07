import torch.nn as nn
import torch.optim as optim
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler
import os
from torchtext.vocab import Vocab

from model.token2token_transformer import Token2TokenTransformer
from preprocess import get_dataset_and_vocab
import tqdm

def train(
        dataset_name="JEC",sufix_exp_folder="exp2_epoch500",
        epoch_num = 500,# 10
        
        # データセット、問題設定
        source_max_length=128, # <eos><bos>含まず
        target_max_length=128, # <eos><bos>含まず
        source_vocab_max_size=8192,# special token 含む
        target_vocab_max_size=8192,# special token 含む
        lr=1e-5,lr_min=1e-7,# lr_minはスケジューラによる最終値
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
    train_dataset,valid_dataset,test_dataset,ja_v,en_v,j_tok,e_tok = get_dataset_and_vocab(source_max_length,target_max_length,source_vocab_max_size,target_vocab_max_size,dataset_name=dataset_name)

    # 学習関係
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=Token2TokenTransformer(
        src_vocab_size=source_vocab_max_size,
        tgt_vocab_size=target_vocab_max_size,
        src_max_len=source_max_length+2, # 前後のtoken
        tgt_max_len=target_max_length+1, # 前後のtoken
        ).to(device)
    print("model loaded")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    valid_dataloader=DataLoader(valid_dataset,batch_size=batch_size,shuffle=False) if valid_dataset!=None else None
    test_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=False) if valid_dataset!=None else None
    scheduler = CosineLRScheduler(optimizer,
                                t_initial=epoch_num*len(train_dataloader), lr_min=lr_min,
                                warmup_t=100, warmup_lr_init=1e-7, warmup_prefix=True) 


    writer = SummaryWriter(exp_dir)
    writer.add_text("meta_data",str(dict(
                dataset_name=dataset_name,sufix_exp_folder=sufix_exp_folder,
        epoch_num = epoch_num,
        # データセット、問題設定
        src_max_length=source_max_length, # <eos><bos>含まず
        tgt_max_length=target_max_length, # <eos><bos>含まず
        src_vocab_max_size=source_vocab_max_size,
        tgt_vocab_max_size=source_vocab_max_size,
        batch_size=batch_size,
        approxmate_totalstep=epoch_num*len(train_dataloader), 
        soruce_sentence_num=len(train_dataset),
        src_vocab_size=len(ja_v),
        tgt_vocab_size=len(en_v),
        lr=lr,
    )))
    step = 0
    train_loss = 0

    print("train start")
    model.set_train_setting(device,criterion,ja_v,en_v,writer)
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
                if valid_dataset!=None and i % 100 ==  0:
                    model.valid_start(epoch,step)
                    for i, data in tqdm.tqdm(enumerate(valid_dataloader)): 
                        loss=model.valid_step(get_val_loss,**data)
                        if get_val_loss:
                            writer.add_scalar("Loss/valid",loss.item(),step)
                    model.valid_end()

    writer.close()
    torch.save(model.state_dict(),os.path.join(exp_dir,"ja2en_middle.pth"))

if __name__=="__main__":
    # train()
    # train(dataset_name="kftt",sufix_exp_folder="exp3_epoch100",epoch_num=100) ex3 は途中終了した。

    # train(dataset_name="kftt_16k",
    #     source_vocab_max_size=16384,
    #     target_vocab_max_size=16384,
    #     sufix_exp_folder="exp4_vocab_16k_epoch10",
    #     epoch_num=10)
    
    # train(dataset_name="kftt_32k",
    #     source_vocab_max_size=32768,
    #     target_vocab_max_size=32768,
    #     sufix_exp_folder="exp4_vocab_32k_epoch10",
    #     epoch_num=10)
    # train(dataset_name="kftt_32k",
    #     source_vocab_max_size=32768,
    #     target_vocab_max_size=32768,
    #     sufix_exp_folder="exp5_vocab_16k_epoch10_batch64_lr1e-4",
        # batch_size=64,  epoch_num=10,lr=1e-4)
  
    # train(dataset_name="kftt",
    #     sufix_exp_folder="debug",
    #     source_max_length=32,
    #     target_max_length=32,
    #     batch_size=128,
    #     epoch_num=10)

    train(dataset_name="jesc",
        sufix_exp_folder="exp1",
        source_max_length=32,
        target_max_length=32,
        source_vocab_max_size=32768,
        target_vocab_max_size=32768,
        batch_size=128,
        lr=1e-4,lr_min=1e-6,
        epoch_num=10)

    
