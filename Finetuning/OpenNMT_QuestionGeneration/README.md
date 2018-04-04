**Evaluating the OpenNMT Model**

- This is the implementation of paper [Learning to Ask: Neural Question Generation for
Reading Comprehension](https://arxiv.org/pdf/1705.00106.pdf).
- Code is adapted from [https://github.com/deepmipt/question\_generation](https://github.com/deepmipt/question_generation)
- Please contact the original authors for questions and suggestions. 

Question generation (QG) aims to create natural questions from a given a sentence or paragraph. One key application of question generation is to generate questions for reading comprehension materials (Heilman and Smith, 2010). Question generation systems can aid in the development of annotated data sets for natural language processing (NLP) research in reading comprehension and question answering.

Here, we introduce an attention-based sequence learning model for the task and investigate the effect of encoding paragraph-level information.

1. **OpenNMT Model**
This model is trainable end-to-end via sequence-to-sequence learning. This model is partially inspired by the way in which a human would solve the task. To ask a natural question, people usually pay attention to certain parts of the input sentence, as well as associating context information from the paragraph. In this model, the conditional probability uses RNN encoder-decoder architecture (Bahdanau et al., 2015; Cho et al., 2014), and adopt the global attention mechanism (Luong et al., 2015a) to make the model focus on certain elements of the input when generating each word during decoding.

2. **OpenNMT Training Dataset**
The OpenNMT model is trained on [Standford Question Answering Dataset (SQUAD)](https://rajpurkar.github.io/SQuAD-explorer/) . SQUAD is a new reading comprehension dataset, consisting of questions posed by crowd-workers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage. With 100,000+ question-answer pairs on 500+ articles, SQuAD is significantly larger than previous reading comprehension datasets. For training this model, we first run Stanford CoreNLP (Manning et al., 2014) for pre-processing -  tokenization and sentence splitting. We then lower-case the entire dataset. With the offset of the answer to each question, we locate the sentence containing the answer and use it as the input sentence.
To train the model, please follow instrucion [here in the original GitHub](https://github.com/antriv/Transfer_Learning_Text/blob/master/Finetuning/OpenNMT_QuestionGeneration/Instructions.md)


3. **OpenNMT Test**
There is a test available in the open GitHub for the trained OpenNMT model ( **Figure 1** ). Here in Figure 1, we can enter any paragraph to test. The trained OpenNMT model selects the best probable span from the paragraph as an answer and generates a query for that answer span. The output file is a TSV, where column 1 is a question, column 2 is an answer and column 2 is a confidence score from the model.

![alt text](https://github.com/antriv/Transfer_Learning_Text/blob/master/Finetuning/OpenNMT_QuestionGeneration/screenshots/opennmt1.PNG)

4. **Creating QA pairs with OpenNMT model for our comparison study**
Instead of trying QA on multiple disjoint small paragraphs, we wanted to generate QA pairs for a big corpus using the trained OpenNMT model. For creating our test corpus, we choose the book [Future Computed](https://msblob.blob.core.windows.net/ncmedia/2018/01/The-Future-Computed.pdf) by Harry Shum and Brad Smith. We converted the online book PDF to a word format and removed all images and diagrams from the book. Our test corpus now consists of text only. We tested the code on a few paragraphs from this corpus ( **Figure**** 2a **and** 2b**).

![alt text](https://github.com/antriv/Transfer_Learning_Text/blob/master/Finetuning/OpenNMT_QuestionGeneration/screenshots/opennmt2a.PNG)

![alt text](https://github.com/antriv/Transfer_Learning_Text/blob/master/Finetuning/OpenNMT_QuestionGeneration/screenshots/opennmt2b.PNG)

5. **Existing Resources**

**Paper:** [https://arxiv.org/pdf/1705.00106.pdf](https://arxiv.org/pdf/1705.00106.pdf)

**GitHub:** [https://github.com/deepmipt/question\_generation](https://github.com/deepmipt/question_generation)
