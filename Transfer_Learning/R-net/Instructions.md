# R-NET: MACHINE READING COMPREHENSION WITH SELF MATCHING NETWORKS

The dataset used for this task is Stanford Question Answering Dataset (https://rajpurkar.github.io/SQuAD-explorer/). Pretrained GloVe embeddings are used for both words (https://nlp.stanford.edu/projects/glove/) and characters (https://github.com/minimaxir/char-embeddings/blob/master/glove.840B.300d-char.txt).

## Requirements
  * Python2.7
  * NumPy
  * tqdm
  * spacy
  * TensorFlow==1.2

# Downloads and Setup
Once you clone this repo, run the following lines from bash **just once** to process the dataset (SQuAD).
```shell
$ pipenv install
$ bash setup.sh
$ pipenv shell
$ python process.py --reduce_glove True --process True
```

# Training / Testing / Debugging / Interactive Demo
You can change the hyperparameters from params.py file to fit the model in your GPU. To train the model, run the following line.
```shell
$ python model.py
```
To test or debug your model after training, change mode="train" to debug or test from params.py file and run the model.

**To use demo, put batch size = 1**

# Tensorboard
Run tensorboard for visualisation.
```shell
$ tensorboard --logdir=r-net:train/
```

