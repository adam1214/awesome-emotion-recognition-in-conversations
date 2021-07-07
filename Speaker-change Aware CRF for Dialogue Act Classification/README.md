### [Paper: Speaker-change Aware CRF for Dialogue Act Classification](https://aclanthology.org/2020.coling-main.40.pdf)
### [Source Code](https://bitbucket.org/guokan_shang/da-classification/src/master/)
## Abstract
* Dialogue Act:對話動作，指的是說話者想從言語中表現的意圖(**communicative intention**)，DA的識別簡化對話語的解釋，能幫助機器理解對話。
* Recent work in Dialogue Act (DA) classification approaches the task as a **sequence labeling problem**, using **NN models coupled with a CRF as the last layer**
* Experiments on the **SwDA corpus** show that our modified CRF layer
* modified CRF layer: taking **speaker-change** into account

## Introduction
* assigning to each utterance a DA label to represent its communicative intention(seq2seq labeling preoblem)
* necessary for a DA classification model to **capture dependencies both at the utterance level and at the label level**
    *  It is difficult to predict the DA of a single utterance without having access to the other utterances in the context
    *  different labels have **different transition probabilities** to other labels
*  Related work: BiLSTM-CRF model (Huang et al., 2015; Lample et al., 2016)
    1.  a bidirectional recurrent neural network with LSTM cells is first applied to capture the dependencies among **consecutive utterances**
    2.   CRF layer is used to capture the dependencies among **consecutive DA labels**
*  linear chain CRF is the most common variant
    *  only neighboring labels are dependent (first-order Markov assumption)
*  While **traditional CRF** requires **defining a potentially large set of handcrafted feature functions** (each weighted with a parameter to be trained), **neural CRF has only two parameterized feature functions (emission and transition)** that are trained with the other parameters of the network in an end-to-end fashion.

## Motivation
* Most sequence labeling tasks(ex: POS tagging) involve only two sequences: input and target. In DA classification, we need **additional input sequence**, that of **speaker-identifiers**(greatly improve DA prediction).
* However, state-of-the-art DA classification models ignore the sequence of speaker-identifiers, so we propose in this paper a **simple modification of the CRF layer where the label transition matrix is conditioned on speaker-change**

## Related work
* we first introduce the two major DA classification approaches
    1. Multi-class classification
        * consecutive DA labels are considered to be **independent**. The DA label of each utterance is predicted in isolation by a classifier such as, e.g., naive Bayes, Maxent, or SVM.    
        * and then some people using NN to DA classification. deep learning has shown promising results even with some simple architectures
        * More recent work developed more advanced models, and started taking into account the dependencies among consecutive utterances
    2. Sequence labeling
        * the DA labels for all the utterances in the conversation are classified together
            * Traditional work: HMM & CRF with handcrafted features
                * HMM: 
                    hidden states(狀態轉移): DA labels (an n-gram language model trained on DA label sequences)
                    observations(已知答案): utts
* focus on previous work involving the use of BiLSTM-CRF and speaker information
    * BiLSTM-CRF
       Kumar et al. (2018) were the first to introduce the BiLSTM-CRF architecture for DA classification.
        *    Two level training:
           1. the text of each utterance is separately encoded by a shared bidirectional LSTM (BiLSTM) with last-pooling, resulting in a sequence of vectors
           2. that sequence is passed through another BiLSTM topped by a CRF layer
       *    Testing:
       the optimal output label sequence is retrieved from the trained model by **Viterbi algorithm**
    * Speaker information
      * Bothe et al. (2018), the utterance representation is the concatenation of the one-hot encoded speaker-identifier, e.g., A as [1, 0] and B as [0, 1], with the output of the RNNbased character-level utterance encoder.
       * Li and Wu (2016) and Liu et al. (2017) choose to concatenate the **speaker-change vector** with the representation obtained via their CNN-based and RNN-based word-level utterance encoders
       * Kalchbrenner and Blunsom (2013) proposed to let the recurrent and output weights of the RNN cell be conditioned on speaker-identifier
       * Stolcke et al. (2000) proposed to train different discourse grammars(話語語法) for different speakers, to guide DA label transitions in HMM.

## Model
![](https://i.imgur.com/DOBZuYp.png)

*    Utterance encoder:Each utterance is separately encoded by a shared forward RNN with LSTM cells. Only the last annotation is retained(last pooling)(每一個utt編碼完只取最後一個部分當成其編碼結果,標示為u^t)
*    BiLSTM layer: The sequence of utterance embeddings is then passed on to a bidirectional LSTM, returning the sequence of **conversation-level utterance representations**(標示為v^t)
*    CRF layer: v^t可以過fully-connected layer再過一層softmax即可，但是若將DA序列視為一個整體，這會產生一個非最佳化的結果。這就是我們採用CRF layer的原因。CRF models the conditional probability P(Y|X) of an entire target sequence Y given an entire input sequence X. Thus, it guarantees an optimal global solution.
        *    ![](https://i.imgur.com/if34BUA.png)
                 sum of emission scores (or state scores) and transition scores over all time steps
        *    ![](https://i.imgur.com/oVvo8mc.png)
                Emission (state) scores是架構圖中虛線部分，中括號內是指element在y^t這個index，不是相乘，所以運算部分不包含中括號，運算值越大代表model在t這個時間點輸出y這個DA label的信心值越大。
        *    ![](https://i.imgur.com/Z4nZUWa.png)
                G是trasition matrix，size為33(因為有label總共有3種)，該matrix在所有時間點都是共享的
        *    The CRF layer is parameterized by W, b, and G.
*    To learn W, b, G in CRF & those of the previous layers, maximum likelihood estimation is used. Our loss:![](https://i.imgur.com/6EOX2Rp.png) (M指的是有M段conversations)
*    Using Viterbi algo. to predict optimal output label sequence