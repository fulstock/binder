rm -rf $HOME/.cache/huggingface/datasets
python $HOME/biencoder/logs/clean.py

python ./train_binder.py