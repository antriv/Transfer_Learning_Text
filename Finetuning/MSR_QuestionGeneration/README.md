**Evaluating the SynNet Model**

Here we go over a technique for finetuning in machine reading comprehension (MRC) using a novel two-stage synthesis network ( **SynNet** ). Given a high-performing MRC model in one domain, our technique aims to answer questions about documents in another domain, where we use no labeled data of question-answer pairs.

1. **The SynNet Model**
The model consists of five different layers as Shown in **Figure 1**.
![alt text](https://github.com/antriv/Transfer_Learning_Text/blob/master/Finetuning/MSR_QuestionGeneration/screenshots/synnet1.PNG)

2. **SynNet Model Layers**
The SynNet is trained to synthesize the answer and the question, given the paragraph. The first stage of the model, an answer synthesis module, uses a bi-directional LSTM to predict IOB tags on the input paragraph, which mark out key semantic concepts that are likely answers. The second stage, a question synthesis module, uses a uni-directional LSTM to generate the question, while attending on embeddings of the words in the paragraph and IOB ids. Although multiple spans in the paragraph could be identified as potential answers, we pick one span when generating the question.

3. **SynNet Training Dataset**
The SynNet model is trained on [Standford Question Answering Dataset (SQUAD)](https://rajpurkar.github.io/SQuAD-explorer/). SQUAD is a new reading comprehension dataset, consisting of questions posed by crowd-workers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage. With 100,000+ question-answer pairs on 500+ articles, SQuAD is significantly larger than previous reading comprehension datasets.

    In this SynNet model, we decompose the process of generating question-answer pairs into two steps:

      a. **Answer Synthesis** - **The answer generation conditioned on the paragraph**
    We generate the answer ﬁrst because answers are usually key semantic concepts, while questions can be viewed as a full sentence composed to inquire about the concept. In this approach, based on the supervised data available in one domain, the SynNet first learns a general pattern of identifying potential &quot;interestingness&quot; in an article. These are key knowledge points, named entities, or semantic concepts that are usually answers that people may ask for.


      b. **Question Synthesis** - **The question generation conditioned on the paragraph and the answer.**
    Then, in the second stage, the model learns to form natural language questions around these potential answers, within the context of the article.

4. **SynNet**  **Test**
Once trained, the SynNet can be applied to a niche domain, read the documents in the new domain and then generate pseudo questions and answers against these documents. Thus, it forms the necessary training data to train an MRC system or an automated answer generation system for that new domain. Two examples of generated questions and answers from NEWSQA articles are illustrated in **Figure 2**.

![alt text](https://github.com/antriv/Transfer_Learning_Text/blob/master/Finetuning/MSR_QuestionGeneration/screenshots/synnet2.PNG)

5. **Creating a QA-Bot with SynNet model for our comparison study**
Instead of trying to generate QA pairs on multiple disjoint small paragraphs, we wanted to create a QA-Bot our Future Computed book corpus. However, the open GitHub code does not allow custom corpuses to be converted to a target corpus. The authors manually created the NewsQA dataset as target corpus for testing. Thus, we could not test the performance of this model on our book test corpus. We are currently working on creating a way to convert our book corpus to a format where SynNet could be tested on this. If we are successful, we will merge the method to the open SynNet GitHub.

6. **Existing Resource:**
        a. Paper: [https://arxiv.org/pdf/1706.09789.pdf](https://arxiv.org/pdf/1706.09789.pdf)
        b. Github: [https://github.com/davidgolub/QuestionGeneration](https://github.com/davidgolub/QuestionGeneration)
