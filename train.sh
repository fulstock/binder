rm -rf $HOME/.cache/huggingface/datasets
python $HOME/biencoder/logs/clean.py

python3.9 ./run_ner.py ./conf/rumodernbert.json