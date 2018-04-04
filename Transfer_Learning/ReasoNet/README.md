**Evaluating the ReasoNet Model**

In this paper by [Shen et. al., 2017](https://arxiv.org/pdf/1609.05284.pdf), the authors describe a novel neural network architecture called the Reasoning Network (ReasoNet) for machine comprehension tasks. The ReasoNet model mimic the inference process of human readers. With a question in mind, ReasoNets read a document repeatedly, each time focusing on different parts of the document until a satisfying answer is found or formed.

ReasoNets make use of multiple turns to effectively exploit and then reason over the relation among queries, documents, and answers. Different from previous approaches using a fixed number of turns during inference, ReasoNets introduce a termination state to relax this constraint on the reasoning depth. With the use of reinforcement learning, ReasoNets can dynamically determine whether to continue the comprehension process after digesting intermediate results, or to terminate reading when it concludes that existing information is adequate to produce an answer.

1. **The ReasoNet Model**

The model consists of five different layers as Shown in **Figure 1**.

![alt text](https://github.com/antriv/Transfer_Learning_Text/blob/master/Transfer_Learning/ReasoNet/screenshots/reasonet1.PNG)


2. **ReasoNet Model Layers**

  a. **Memory**

  The external memory is denoted as M. It is a list of word consisting of fixed dimensional vector.

  b. **Attention**

  An attention vector is generated based on the current internal state and the external memory.

  c. **Internal State**

  The internal state is denoted as a vector representation of the question state. Typically, the initial state is the last-word vector   representation of query by an RNN. The sequence of internal states is modeled by an RNN.

  d. **Termination Gate**

  The termination gate generates a binary random variable according to the current internal state. If the random variable value is true, the ReasoNet stops, and the answer module executes, otherwise the ReasoNet generates an attention vector and feeds the vector into the state network to update the next internal state.

  e. **Answer**

  The action of answer module is triggered when the termination gate variable is true
  
  

3. **ReasoNet Training Dataset**

There are three training datasets for the ReasoNet model. We can use any of them to train the model.

  a. ReasoNet is trained on [Standford Question Answering Dataset (SQUAD)](https://rajpurkar.github.io/SQuAD-explorer/) as just as the BiDAF model described above. Please refer above for more SQUAD dataset details.
  
  b. ReasoNet is trained on [CNN and Daily Mail Datasets](https://cs.nyu.edu/~kcho/DMQA/). Hermann et al. (2015) seek to solve this problem by creating over a million training examples by pairing CNN and Daily Mail news articles with their summarized bullet points and show that a neural network can then be trained to give good performance on this task. Each dataset contains many documents (90k and 197k each), and each document companies on average 4 questions approximately. Each question is a sentence with one missing word/phrase which can be found from the accompanying document/context.
  
  c. Recent analysis and results on the cloze-style machine comprehension tasks have suggested some simple models without multiturn reasoning can achieve reasonable performance. Based on these results, a synthetic structured Graph Reachability dataset is constructed to evaluate longer range machine inference and reasoning capability. This dataset allows ReasoNets to have the capability to handle long range relationships. Here we have 2 two synthetic datasets -
  
   i. A small graph dataset containing 500K small graphs, where each graph contains 9 nodes and 16 direct edges to randomly connect pairs of nodes.
      
   ii. A large graph dataset containing 500K graphs, where each graph contains 18 nodes and 32 random direct edges.



4. **ReasoNet Test**

The ReasoNet paper authors provided us with a SQUAD-trained ReasoNet model and some sample demo code ( **Figure 2** ).

![alt text](https://github.com/antriv/Transfer_Learning_Text/blob/master/Transfer_Learning/ReasoNet/screenshots/reasonet2.PNG)


5. **Creating a QA-Bot with SynNet model for our comparison study**

Instead of trying to generate answers on multiple disjoint small paragraphs, we wanted to create a QA-Bot our Future Computed book corpus. However, the ReasoNet model in its current form does not work for a large corpus. Given a larger paragraph, this model usually takes a very long time and comes back with a probable span as an answer which might not make any sense at all.


6. **Existing Resources**

**Paper:** [https://arxiv.org/pdf/1609.05284.pdf](https://arxiv.org/pdf/1609.05284.pdf)

**GitHub:** No public GitHub code available
