from pandas import Series
from collections.abc import Callable
from collections import Counter
from pandarallel import pandarallel
from torchtext.vocab import vocab,Vocab

pandarallel.initialize()

def create_vocab(series:Series,tokenizer:Callable[[str],list[str]],
                 vocab_max_size:int)->tuple[Vocab,dict]:
    """
    現在はvocab_max_size以外の絞り込みはない。
    例えば、未実装だが、最低出現個数以上を採用などもある。
    """
    info_dict={}
    print("tokenize")
    tokenized_texts = series.parallel_apply(tokenizer)
    print("tokenize end")
    print(tokenized_texts[0]) # for debug
    __tmp = [] # 全てのtokenリスト
    for i in range(len(tokenized_texts)):
        __tmp.extend(tokenized_texts[i])
    counter=Counter()
    counter.update(__tmp)
    info_dict.update(
        sentence_num=len(tokenized_texts),
        original_token_total_num=len(__tmp),
        original_vocab_size=len(counter)
    )
    specials=['<unk>', '<pad>', '<bos>', '<eos>']
    counter=Counter(dict(counter.most_common(vocab_max_size-len(specials))))
    v=vocab(counter,specials=(specials))
    v.set_default_index(v['<unk>'])
    print("vocab created")
    info_dict.update(
        cover_token_num=counter.total(),
        create_vocab_size=len(v))
    return v,info_dict
