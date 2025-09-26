rm -rf $HOME/.cache/huggingface/datasets
python $HOME/biencoder/logs/clean.py

python3.10 ./run_ner.py ./conf/train-ruroberta.json