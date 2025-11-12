rm -rf $HOME/.cache/huggingface/datasets
python $HOME/biencoder/logs/clean.py

#### track1
### kw
python3.9 ./run_ner.py ./conf/rutermeval/track1/kw/flat-pure.json
python3.9 ./run_ner.py ./conf/rutermeval/track1/kw/full.json
python3.9 ./run_ner.py ./conf/rutermeval/track1/kw/winc.json
python3.9 ./run_ner.py ./conf/rutermeval/track1/kw/lemwinc.json

rm -rf $HOME/.cache/huggingface/datasets
python $HOME/biencoder/logs/clean.py

#### track2
### kw
python3.9 ./run_ner.py ./conf/rutermeval/track2/kw/flat-pure.json
python3.9 ./run_ner.py ./conf/rutermeval/track2/kw/full.json
python3.9 ./run_ner.py ./conf/rutermeval/track2/kw/winc.json
python3.9 ./run_ner.py ./conf/rutermeval/track2/kw/lemwinc.json

rm -rf $HOME/.cache/huggingface/datasets
python $HOME/biencoder/logs/clean.py

#### track3
### kw
python3.9 ./run_ner.py ./conf/rutermeval/track3/kw/flat-pure.json
python3.9 ./run_ner.py ./conf/rutermeval/track3/kw/full.json
python3.9 ./run_ner.py ./conf/rutermeval/track3/kw/winc.json
python3.9 ./run_ner.py ./conf/rutermeval/track3/kw/lemwinc.json

rm -rf $HOME/.cache/huggingface/datasets
python $HOME/biencoder/logs/clean.py

python3.9 ./run_ner.py ./conf/rutermeval/track1/kw/lemwincdamage.json

python3.9 ./run_ner.py ./conf/rutermeval/track2/kw/lemwincdamage.json

python3.9 ./run_ner.py ./conf/rutermeval/track3/kw/lemwincdamage.json