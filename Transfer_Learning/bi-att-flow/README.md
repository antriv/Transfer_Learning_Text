**Evaluating the Bi-Directional Attention Flow (BIDAF) Model**

To accurately answer questions based on a document, we need to be able to model complex interactions between the document and query. This paper by [Seo et. al., 2017](https://arxiv.org/pdf/1611.01603.pdf) uses the Bi-Directional Attention Flow (BiDAF) network to process an attention context that is query aware and represents the document at different levels of granularity. Its main advantage is the bidirectional attention flow, which eliminates the early summarization issue we see with normal uni-directional attentional interfaces. Furthermore, the BiDAF does not use just one context vector (i.e, the summary) but uses vectors from all time steps.

1. **BiDAF Model**

The model consists of six different layers as shown in **Figure 1**.

2. **BiDAF Model Layers**

**a.**** Character-Level Embedding**

A char-CNN used to map each word to an embedding.

**b.**** Word-Level Embedding**

Pretrained GLOVE embeddings are used to map each word to a vector.

**c.**** Phrase-Level Embedding**

Uses char-level and word-level embedding layers to refine the embeddings of words using context from other words.

**d.**** Attentional Interface**

Uses the query and document vectors from previous three layers to produce query aware feature vectors for each word in the document. Basically, we are trying to create a higher representation for each word in the document that is query aware, so we can get the right answer.

**e.**** Modeling Layer**

Uses an RNN to scan the document.

**f.**** Output Layer**

Gives us the answer to the query.

1. **BiDAF Training Dataset**

The BiDAF model is trained on [Standford Question Answering Dataset (SQUAD)](https://rajpurkar.github.io/SQuAD-explorer/). SQUAD is a new reading comprehension dataset, consisting of questions posed by crowd-workers on a set of Wikipedia articles, where the answer to every question is a segment of text, or _span_, from the corresponding reading passage. With 100,000+ question-answer pairs on 500+ articles, SQuAD is significantly larger than previous reading comprehension datasets.

1. **BiDAF Test**

There is a public [BiDAF Demo Link](http://35.165.153.16:1995/) available for testing the trained BiDAF model ( **Figure 2** ). Here in Figure 3, we can enter any paragraph to test. In the question section, we can add a question we want to ask about the paragraph on the left. The trained BiDAF model selects the best probable span from the paragraph as an answer to the query we put in and this span is displayed in the answer section.

1. **Creating a QA-Bot with BiDAF model for our comparison study**

Instead of trying QA on multiple disjoint small paragraphs, we wanted to create a QA-Bot for a big corpus using the trained BiDAF model. For creating our test corpus, we choose the book [Future Computed](https://msblob.blob.core.windows.net/ncmedia/2018/01/The-Future-Computed.pdf) by Harry Shum and Brad Smith. We converted the online book PDF to a word format and removed all images and diagrams from the book. Our test corpus now consists of text only. We wrote a bot script where we use this corpus only for testing any question coming from the bot UI. We operationalized the bot and tested it with several questions on the topic of Artificial Intelligence (AI) ( **Figure**** 3**).

1. **Existing Resources**

**Paper:** [https://arxiv.org/pdf/1611.01603.pdf](https://arxiv.org/pdf/1611.01603.pdf)

**GitHub:** [https://github.com/allenai/bi-att-flow](https://github.com/allenai/bi-att-flow)

**Demo Link:** [http://allgood.cs.washington.edu:1995/](http://allgood.cs.washington.edu:1995/)

1. **Our Contribution**  **-**   [https://github.com/antriv/bi-att-flow](https://github.com/antriv/bi-att-flow)

1. We have converted code to TF 1.2 v from TF 0.12 v.
2. We have the demo link working with latest TF 1.2v
3. We wrote script for testing any custom paragraph - 1\_comprehend.py
4. We wrote script for testing any static paragraph- 2\_comprehend\_static.py
5. We wrote a flask api server to operationalize the model locally on a Linux DSVM- 3\_comprehend\_future\_computed\_run\_flask\_server\_local.py
6. We wrote a flask api client to test the model locally on a Linux DSVM - 4\_comprehend\_future\_computed\_request\_local.py
7. Created a bot backend with model- 5\_comprehend\_future\_computed\_run\_flask\_server\_bot.py
8. We have a bot working with this model as the backend. ( **Figure 3** )
