# Bi-directional Attention Flow for Machine Comprehension
 
- This the original implementation of [Bi-directional Attention Flow for Machine Comprehension][paper] (Seo et al., 2016).
- This is tensorflow v1.2.0 comaptible version.
- Refer to [the original paper][paper] for more details.
- See [SQuAD Leaderboard][squad] to compare with other models.
- Please contact the original authors [Minjoon Seo][minjoon] ([@seominjoon][minjoon-github]) for questions and suggestions. 

## 0. Requirements
#### General
- Python 3
- unzip

#### Python Packages
- tensorflow (deep learning library, verified on 1.2.0)
- nltk
- tqdm


## 1. Pre-processing
First, prepare data. Donwload SQuAD data and GloVe and nltk corpus
(~850 MB, this will download files to `$HOME/data`):
```
chmod +x download.sh; ./download.sh
```

Second, Preprocess Stanford QA dataset (along with GloVe vectors) and save them in `$PWD/data/squad` (~5 minutes):
```
python -m squad.prepro
```

## 2. Training
The model was trained with 4 NVidia Tesla K80 GPUs, CUDA & cuDNN.If your GPU RAM does not have enough memory, you can either decrease batch size (performance might degrade), or you can use multi GPU (see below).
The training converges at ~18k steps, and it took ~10s per step (i.e. ~50 hours).

Before training, it is recommended to first try the following code to verify everything is okay and memory is sufficient:
```
python -m basic.cli --mode train --noload --debug
```

Then to fully train, run:
```
python -m basic.cli --mode train --noload
```

You can speed up the training process with optimization flags:
```
python -m basic.cli --mode train --noload --len_opt --cluster
```
You can still omit them, but training will be much slower.


## 3. Test
To test, run:
```
python -m basic.cli
```

Similarly to training, you can give the optimization flags to speed up test (5 minutes on dev data):
```
python -m basic.cli --len_opt --cluster
```

This command loads the most recently saved model during training and begins testing on the test data.
After the process ends, it prints F1 and EM scores, and also outputs a json file (`$PWD/out/basic/00/answer/test-####.json`,
where `####` is the step # that the model was saved).
Note that the printed scores are not official (our scoring scheme is a bit harsher).
To obtain the official number, use the official evaluator (copied in `squad` folder) and the output json file:

```
python squad/evaluate-v1.1.py $HOME/data/squad/dev-v1.1.json out/basic/00/answer/test-####.json
```


## 4. Multi-GPU Training & Testing
This model supports multi-GPU training.
If you want to use batch size of 60 (default) but if you have 3 GPUs with 4GB of RAM,
then you initialize each GPU with batch size of 20, and combine the gradients on CPU.
This can be easily done by running:
```
python -m basic.cli --mode train --noload --num_gpus 3 --batch_size 20
```

Similarly, you can speed up your testing by:
```
python -m basic.cli --num_gpus 3 --batch_size 20 
```

## 5. Demo
You may try your own paragraph and question by pasting your paragraph in the file `SAMPLE_PARAGRAPH` and then running
```
python 1_comprehend.py SAMPLE_PARAGRAPH <question>
```

## 6. Demo on static document
You may try any question by keeping the paragraph content constant. 
```
python 2_comprehend_static.py <question>
```

## 7. Bot Experience on static document
You may create a bot using your own paragraph/document. Here we use Harry Shum's Book "Future Computed" as out static document. 
Now, we need to operationalize the model on this document. We use python Flask API to operationalize the model locally.
```
python 3_comprehend_future_computed_run_flask_server.py
```
This operationalizes the model at port 5000. To test the bot locally we can run:
```
python 4_comprehend_future_computed_request.py "What is the future of AI"?
```

[multi-gpu]: https://www.tensorflow.org/versions/r0.11/tutorials/deep_cnn/index.html#training-a-model-using-multiple-gpu-cards
[squad]: http://stanford-qa.com
[paper]: https://arxiv.org/abs/1611.01603
[worksheet]: https://worksheets.codalab.org/worksheets/0x37a9b8c44f6845c28866267ef941c89d/
[minjoon]: https://seominjoon.github.io
[minjoon-github]: https://github.com/seominjoon
[v0.2.1]: https://github.com/allenai/bi-att-flow/tree/v0.2.1
