# JaSPLADE
SPLADEの実験を回すためのコード
日本語でも英語でも可能
pyseriniを使う推論と違ってfloatで保存するのでインデックスもデカくならないし，速いと思います


shell_exampleのコードの部分に適切な値を入れれば動くと思う
ここら辺を入れとけば動く
```
numba
h5py
torch
transformers
pytrec_eval
pyserini
```

日本語で動かすなら
'''
!pip install fugashi ipadic unidic-lite
'''

入力データ（--local_dataで自前のデータを入力可能）
```
tsv: クエリid クエリ
json, jsonl: query_id, query をそれぞれkeyとする

tsv: 文書id 文書 タイトル
json, jsonl: id, contents, title をそれぞれkeyとする
```


