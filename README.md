# Seq2Seq
このディレクトリは機械翻訳や音声認識などの比較的小さめなモデルに関しての遊び場である。

# 流れ

## データダウンロード
任意の場所にデータを入れます。

## データの事前処理
preprocessed.pyでは、data_nameに紐づいて事前処理を行い、保存します。
新たにデータセットを加えた場合や、語彙サイズなどの変更を行いたい場合などは、新しいデータセット名をプログラム中に定義してください。
辞書作成などは全て学習データのみから作成されます。

## データの学習
現在はtoken2tokenのみサポートしています。
ここでのロスは1ステップごとの表示です。

# 今後の予定
- testデータへの対応
- 音声2テキストの対応
- 学習済みモデルへの対応
- beam searchへの対応
- 音声2音声モデルへの対応
- テキスト2テキストモデルへの対応
- 大規模なデータセットにおけるIterable Dataset
    - https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable
    - https://huggingface.co/docs/datasets/stream
    - https://huggingface.co/docs/datasets/v1.10.1/dataset_streaming.html

# 参考
JESCコーパスのtrainデータをして日英翻訳を行った時のdevセット上でのBLUEスコアは、8程度。
```
    train(dataset_name="jesc",
        sufix_exp_folder="exp1",
        source_max_length=32,
        target_max_length=32,
        source_vocab_max_size=32768,
        target_vocab_max_size=32768,
        batch_size=128,
        lr=1e-4,lr_min=1e-6,
        epoch_num=10)
```
動作環境は
GPU : RTX 4880 (使用メモリはおよそ10G)です。
学習時間は 18時間(目安)。validationの頻度が多い気がするので、下げれば数時間は早くなると思います。

![Image](https://github.com/series2/seq2seq/blob/main/image.png)


# 参考
cl-tohoku base はvocab size 32000
bert base は 30522 (WordPieceのため)