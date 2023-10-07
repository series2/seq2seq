import torch.nn as nn
import torch.optim as optim
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler
from torchmetrics.functional.text import bleu_score
import os
from torchtext.vocab import Vocab

from token2token_transformer import Token2TokenTransformer
from preprocess import get_dataset_and_vocab
import tqdm

# TODO
# deviceの変更
# evaluationの時のmodelのモード変更

def eval_print(writer:SummaryWriter,step:int,model:Token2TokenTransformer,
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

            model.eval()
            with torch.no_grad():
                if valid_dataset!=None and i % 100 ==  0:
                    sents_gen=[] # space区切りのデコード結果
                    tgt_gold=[]
                    for i, data in tqdm.tqdm(enumerate(valid_dataloader)): 
                        src_token, tgt_token = data["src_token"].to(device), data["tgt_token"].to(device)
                        # それぞれ (B,L)

                        if get_val_loss:
                            outputs= model(src_token=src_token, tgt_token=tgt_token[:,:-1],tgt_is_causal=True,)

                            target = nn.functional.one_hot(tgt_token[:,1:], target_vocab_max_size).to(torch.float32)
                            loss = criterion(
                                    torch.reshape(outputs,(-1,target_vocab_max_size)),
                                    torch.reshape(target,(-1,target_vocab_max_size))
                                )
                            writer.add_scalar("Loss/valid",loss.item(),step)
                        
                        gen=model.generate(src_token,en_v['<bos>'],en_v['<eos>']).tolist()
                        sent_wakati=[filter(lambda x: (x!="<pad>" and x!="<bos>" and x!="<eos>"), en_v.lookup_tokens(sent)) for sent in gen]
                        sents=[" ".join(sent) for sent in sent_wakati]
                        sents_gen.extend(sents)

                        gold_ge=tgt_token.tolist()
                        # print(gold_ge)
                        gold_wakati=[filter(lambda x: (x!="<pad>" and x!="<bos>" and x!="<eos>"), en_v.lookup_tokens(sent)) for sent in gold_ge]
                        sents=[[" ".join(sent)] for sent in gold_wakati]
                        tgt_gold.extend(sents)

                        # goldでは元の文を使うべきかunkこみにすべきか
                        # 言語処理の観点からは元の文を使うべきだが、同じ条件下でモデル性能を比較したいなら、unkコミでも大丈夫なはず。ただし、<bos><eos><pad>はそれなりに数が多くて何も考えていなくても成果になってしまう可能性があるので、抜いておく。
                    # TODO paddingなど考えるのがめんどいので、全てのトークンで考える。
                    score=bleu_score(sents_gen,tgt_gold)
                    print(score,sents_gen[0],tgt_gold[0])
                    writer.add_scalar("BLEU/valid",score,step)





                if i % 10 ==  0:
                    print(f"epoch:{epoch+1}  index:{i+1}  loss:{train_loss/10:.10f}")
                    train_loss = 0
                    

                if step%100 == 0:
                    print(f"out_log[step:{step}]")
                    eval_print(writer,step,model,
                               src_token,tgt_token,outputs,ja_v,en_v)


    writer.close()
    torch.save(model.state_dict(),os.path.join(exp_dir,"ja2en_middle.pth"))

    jas=["今日はいい天気ですね。","友達とご飯を食べにいく。"]
    jas_wakatis=[j_tok(el) for el in jas]
    jas_ids=train_dataset.src_data.transform(jas_wakatis)
    
    sents_tokens=model.generate(jas_ids,en_v['<bos>'],en_v['<eos>']).tolist()
    sent_wakati=[filter(lambda x: x!="<pad>", en_v.lookup_tokens(sent)) for sent in sents_tokens]
    sents=[" ".join(sent) for sent in sent_wakati]
    print("\n".join(sents))
    print("end")

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

    
