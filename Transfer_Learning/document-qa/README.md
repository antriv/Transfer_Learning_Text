**Evaluating the Document-QA Model**

This model, by [Clark et. al., 2017](https://arxiv.org/pdf/1710.10723.pdf), considers the problem of adapting neural paragraph-level question answering models to the case where entire documents are given as input. Here the authors proposed a solution that trains models to produce well calibrated confidence scores for their results on individual paragraphs. This model sample multiple paragraphs from the documents during training and use a shared normalization training objective that encourages the model to produce globally correct output. Next, this method is combined with a state-of-the-art pipeline for training models on document QA data.

1. **Document-QA Model**

The model consists of six different layers as shown in **Figure 1**.

2. **Document-QA Model Layers**

    a. **Embedding**

    Embed words using pretrained word vectors. Also embed the characters in each word into size 20 vectors which are learned. Then run a convolution neural network followed by max-pooling to get character-derived embeddings for each word. The character-level and word-level embeddings are then concatenated and passed to the next layer. The word embeddings are not updated during training.

    b. **Pre-Process**

    A shared bi-directional GRU, as described by  [Cho et al., 2014](https://arxiv.org/pdf/1406.1078.pdf), is used to map the question and passage embeddings to context-aware embeddings.

    c. **Attention**

    The BiDAF model, by  [Seo et al., 2016](file:///h), is used to build a query-aware context representation.

    d. **Self-Attention**

    Uses a layer of residual self-attention. The input is passed through another bi-directional GRU. Then we apply the same attention mechanism, only now between the passage and itself. In this case we do not use query-to-context attention. As before, we pass the concatenated output through a linear layer with ReLU activations. This layer is applied residually, so this output is additionally summed with the input.

    e. **Prediction**

    In the last layer of this model a bidirectional GRU is applied, followed by a linear layer that computes answer-start scores for each token. The hidden states of that layer are concatenated with the input and fed into a second bidirectional GRU and linear layer to predict answer-end scores. The SoftMax operation is applied to the start and end scores to produce start and end probabilities, and we optimize the negative loglikelihood of selecting correct start and end tokens.

    f. **Dropout**

    This model also employs variational dropout, where a randomly selected set of hidden units are set to zero across all time steps during training. We dropout the input to all the GRUs, including the word embeddings, as well as the input to the attention mechanisms.

3. **Document-QA **** Training Dataset**

There are two training datasets for the Document-QA model. We can use either of them to train the model.

**a.** Document-QA is trained on [Standford Question Answering Dataset (SQUAD)](https://rajpurkar.github.io/SQuAD-explorer/) as just as the BiDAF model described above. Please refer above sections for more SQUAD dataset details.

**b.** Document-QA is trained on [TriviaQA](http://nlp.cs.washington.edu/triviaqa/) by [Joshi et. Al., 2017](https://arxiv.org/pdf/1705.03551.pdf). TriviaQA is a challenging reading comprehension dataset containing over 650K question-answer-evidence triples. TriviaQA includes 95K question-answer pairs authored by trivia enthusiasts and independently gathered evidence documents, six per question on average, that provide high quality distant supervision for answering the questions. In comparison to other recently introduced large-scale datasets, TriviaQA has –
  - Relatively complex, compositional questions
  - Considerable syntactic and lexical variability between questions and                                       corresponding answer-evidence sentences
  - Requires more cross sentence reasoning to find answers

4. **Document-QA Test**
There is a public [Document-QA Demo Link](https://documentqa.allenai.org/) available for testing the TriviaQA-trained Document-QA model. In this demo, you can ask a question to the web documents by choosing the web option, or to some paragraph by choosing the text option, or to some document by choosing the document option.

5. **Creating a QA-Bot with TriviaQA-trained Document-QA model for our comparison study**
Instead of trying QA on multiple disjoint documents, we wanted to create a QA-Bot for a big corpus using the TriviaQA-trained Document-QA model. For creating our test corpus, we choose the book [Future Computed](https://msblob.blob.core.windows.net/ncmedia/2018/01/The-Future-Computed.pdf) by Harry Shum and Brad Smith. We converted the online book PDF to a word format and removed all images and diagrams from the book. Our test corpus now consists of text only. We wrote a bot script where we use this corpus only for testing any question coming from the bot UI. We operationalized the bot and tested it with several questions on the topic of Artificial Intelligence (AI).

6. **Existing Resources**

  **Paper:** [https://arxiv.org/pdf/1710.10723.pdf](https://arxiv.org/pdf/1710.10723.pdf)

  **GitHub:** [https://github.com/allenai/document-qa](https://github.com/allenai/document-qa)

  **Demo Link:** [https://documentqa.allenai.org/](https://documentqa.allenai.org/)

1. **7.**** Our Contribution**

1. We wrote test script for testing static documents - 2\_run\_on\_static\_documents.py
2. We wrote a flask api server to operationalize the model locally on a Linux DSVM -  3\_run\_flask\_api\_local\_on\_static\_documents.py
3. We wrote a flask api client to test the model locally on a Linux DSVM - 4\_local\_static\_request.py
4. Created a bot backend with model - 5\_run\_flask\_api\_bot\_on\_static\_documents.py
5. We have a bot working with this model as the backend ( **Figure 2** ).
6. We have the demo working comparing the [open TAGME API](https://tagme.d4science.org/tagme/) and Document-QA models ( **Figure 3** ).
7. We have a Web option where we run a query using the TAGME API ( **Figure 4** and **6** ).
8. We have a Document option where we run a query using the TriviaQA-trained Document-QA model on Future Computed book ( **Figure 5** and **7** ).
9. We see that TAGME API works better for generic questions (like- &quot;How will AI affect jobs&quot;). However, the TriviaQA-trained Document-QA model works better on targeted question from a document (like- &quot;What is AI Law&quot;)
