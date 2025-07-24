rm -rf $HOME/.cache/huggingface/datasets
# python $HOME/biencoder/logs/clean.py

python ./run_ner.py ./conf/nerel-attack.json
