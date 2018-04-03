#!/usr/bin/env bash

DATA_DIR=$HOME/data
mkdir $DATA_DIR

# Download SQuAD
SQUAD_DIR=$DATA_DIR/squad
mkdir $SQUAD_DIR
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $SQUAD_DIR/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $SQUAD_DIR/dev-v1.1.json

#Download Models
wget https://www.dropbox.com/s/iahghqnhd8ee4rs/data.zip?dl=1 -O data.zip
unzip data.zip
wget https://www.dropbox.com/s/kqfipzlhkfxhguv/out.zip?dl=1 -O out.zip
unzip out.zip

# Download GloVe
GLOVE_DIR=$DATA_DIR/glove
mkdir $GLOVE_DIR
wget http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR/glove.6B.zip
unzip $GLOVE_DIR/glove.6B.zip -d $GLOVE_DIR

# Download NLTK (for tokenizer)
# Make sure that nltk is installed!
python3 -m nltk.downloader -d $HOME/nltk_data punkt
