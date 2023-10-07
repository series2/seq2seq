import torch
import os

from preprocess import get_dataset_and_vocab
from model.token2token_transformer import Token2TokenTransformer

dataset_name="kftt_16k"
dataset,_,_,j_v,e_v,j_t,e_t=get_dataset_and_vocab(128,128,32768,32768,dataset_name)
sufix_exp_folder="exp5_vocab_16k_epoch10_batch64_lr1e-4"

exp_dir=f"runs/middle/{dataset_name}/{sufix_exp_folder}"

model=Token2TokenTransformer(
    src_vocab_size=32768,
    tgt_vocab_size=32768,
    src_max_len=130,
    tgt_max_len=129,
)
model.load_state_dict(
torch.load(os.path.join(exp_dir,"ja2en_middle.pth")))
model=model.cuda()
model.eval()

# 複数の文の操作
jas=["今日はいい天気ですね。","昼ごはんはおいしかった。"]
jas_wakatis=[j_t(el) for el in jas]
print(jas_wakatis)
# 何と、並列かするには全ての分書の長さが同じでないといけない。。。<pad>を入れるしかなさそう。
jas_ids=torch.stack([dataset.src_data.transform(jas_wakati) for jas_wakati in jas_wakatis])
sents_tokens=model.generate(jas_ids,e_v['<bos>'],e_v['<eos>']).tolist()
sent_wakati=[filter(lambda x: x!="<pad>", e_v.lookup_tokens(sent)) for sent in sents_tokens]
sents=[" ".join(sent) for sent in sent_wakati]
print("- " + "\n\n- ".join(sents))
print("end")

# インタラクティブ
while True:
    ja=input("input japanese(exit:q):")
    # ja="今日はいい天気ですね。"
    if ja=="q":
        break
    ja_wakati=j_t(ja)
    ja_ids=dataset.src_data.transform([ja_wakati,ja_wakati]).squeeze().cuda()
    with torch.no_grad():
        en_pred_text=" ".join(filter(lambda x: x!="<pad>" ,e_v.lookup_tokens(model.generate(ja_ids,e_v['<bos>'],e_v['<eos>'])[0].tolist()))) 
    print(en_pred_text)
    print("end")