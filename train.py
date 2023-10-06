import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler
import os

from token2token_transformer import Token2TokenTransformer
from preprocess import get_dataset_and_vocab

# TODO
# deviceの変更
# evaluationの時のmodelのモード変更


def train(
        dataset_name="JEC",sufix_exp_folder="exp2_epoch500",
        epoch_num = 500,# 10
        
        # データセット、問題設定
        source_max_length=128, # <eos><bos>含まず
        target_max_length=128, # <eos><bos>含まず
        source_vocab_max_size=8192,
        target_vocab_max_size=8192,

        batch_size=32,
        ):
    exp_dir=f"runs/middle/{dataset_name}/{sufix_exp_folder}"
    if os.path.exists(exp_dir):
        l=input(f"folder {exp_dir} exists. Do you want to overwrite it? (y/n):")
        if l=="y" or l=="yes" or l =="Y" :
            print("overwrite.")
        else:
            raise Exception("上書きしないので終了します。")
    print("preprocess start")
    train_dataset,ja_v,en_v,j_tok,e_tok = get_dataset_and_vocab(source_max_length,target_max_length,source_vocab_max_size,target_vocab_max_size,dataset_name=dataset_name)

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
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    scheduler = CosineLRScheduler(optimizer,
                                t_initial=epoch_num*len(train_dataloader), lr_min=1e-7,
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
        soruce_sentence_num=len(train_dataset),
        src_vocab_size=len(ja_v),
        tgt_vocab_size=len(en_v),
    )))
    step = 0
    train_loss = 0

    print("train start")
    for epoch in range(epoch_num):
        model.train()
        for i, data in enumerate(train_dataloader):
            step+=1
            optimizer.zero_grad()
            src_token, tgt_token = data["src_token"].to(device), data["tgt_token"].to(device)
            # それぞれ (B,L)

            outputs= model(src_token=src_token, tgt_token=tgt_token[:,:-1],tgt_is_causal=True,)

            target = nn.functional.one_hot(tgt_token[:,1:], target_vocab_max_size).to(torch.float32)
            # sotより後ろを予測するので。
            # loss = criterion(outputs, target)
            loss = criterion(
                    torch.reshape(outputs,(-1,target_vocab_max_size)),
                    torch.reshape(target,(-1,target_vocab_max_size))
                )
            loss.backward()
            optimizer.step()
            scheduler.step(step)

        
            train_loss += loss.item()
            writer.add_scalar("Loss/train",loss.item(),step)
            writer.add_scalar("lr",scheduler._get_lr(step)[0],step)# 要素ひとつしかない。

            if i % 10 ==  0:
                print(f"epoch:{epoch+1}  index:{i+1}  loss:{train_loss/10:.10f}")
                train_loss = 0

            if step%100 == 0:
                print(f"out_log[step:{step}]")
                src_text=" ".join(filter(lambda x: x!="<pad>" ,ja_v.lookup_tokens(src_token[-1,:].tolist()))) 
                writer.add_text("src",src_text,step)
                print("src :",src_text)
                pred_text=" ".join(filter(lambda x: x!="<pad>" ,en_v.lookup_tokens(outputs[-1,:].argmax(-1).tolist()))) 
                writer.add_text("pred",pred_text,step)
                # print(outputs[-1,:].argmax(-1).tolist())
                # print(outputs)
                # print(target)
                print("pred :",pred_text)
                gen_text=" ".join(filter(lambda x: x!="<pad>" ,en_v.lookup_tokens(model.generate(src_token[-1,:],en_v['<bos>'],en_v['<eos>'])[0].tolist()))) 
                writer.add_text("generate",gen_text,step)
                print("gen :",gen_text)
                gold_text=" ".join(filter(lambda x: x!="<pad>" ,en_v.lookup_tokens(tgt_token[-1,1:].tolist()))) 
                writer.add_text("gold",gold_text,step)
                print("gold :" , gold_text)
                #   print("\n".join([
                #     " ".join(en_v.lookup_tokens(_t)) for _t in outputs.argmax(-1).tolist()
                #     ]))


    writer.close()
    torch.save(model.state_dict(),os.path.join(exp_dir,"ja2en_middle.pth"))

    ja="今日はいい天気ですね。"
    ja_wakati=j_tok(ja)
    ja_ids=train_dataset.j_text_transform([ja_wakati]).squeeze()
    en_pred_text=" ".join(filter(lambda x: x!="<pad>" ,en_v.lookup_tokens(model.generate(ja_ids,en_v['<bos>'],en_v['<eos>'])[0].tolist()))) 
    print(en_pred_text)
    print("end")

if __name__=="__main__":
    # train()
    train(dataset_name="kftt",sufix_exp_folder="exp3_epoch100",epoch_num=100)
    
