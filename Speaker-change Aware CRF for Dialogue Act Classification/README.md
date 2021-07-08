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
                G是trasition matrix，size為KxK(因為有label總共有K種)，該matrix在所有時間點都是共享的
        *    The CRF layer is parameterized by W, b, and G.
*    To learn W, b, G in CRF & those of the previous layers, maximum likelihood estimation is used. Our loss:![](https://i.imgur.com/6EOX2Rp.png) (M指的是有M段conversations)
*    Using Viterbi algo. to predict optimal output label sequence

## Our contribution
*    extend the original CRF so that it considers as additional input, the sequence Z(語者有無改變). That is, **CRF now models P(Y|X,Z) instead of just P(Y|X)**.
*    X: utterance sequence 
     Z: speaker-change sequence (由0 & 1組成，若語者於t與t+1時間點不同，則Z^(t,t+1) = 1)
     Y: DA label sequence
*    transition scores in our modified CRF layer:![](https://i.imgur.com/uAliioU.png)
     G0與G1都是K x K的狀態轉移矩陣，分別對應語者無改變 & 語者有改變的狀態轉移矩陣

## Experimental setup
*    SwDA dataset
        *    telephone conversations recorded between two randomly selected speakers talking about one of various general topics (air pollution, music, football, etc.)
        *    42個互斥DA labels(based on the SWBD-DAMSL annotation scheme)，**imbalanced & long-tailed distribution** 
                ![](https://i.imgur.com/faFpcOR.png)

        *    training set: 1003 conversations
             validation set: 112 conversations
             testing set: 19 conversations
        *    '+'這個label占比8.1%，但不屬42類中，是指語者說該句utt時被別位語者打斷的標記，許多work不討論該label，不過此篇paper有考慮。做法是將被打斷的utt做重新拼接(utt_A+utt_B)，並放在原本utt_A處
*    Implementation and training details
        *    過濾掉不流暢標記，並將字母都轉換成小寫
        *    0.2 dropout was applied to the **utterance embeddings** and **conversation-level utterance representations**, and all LSTM layers had 300 hidden units
        *    The embedding layer was initialized using 300-dimensional word vectors pre-trained with the gensim (Sojka, 2010) implementation of word2vec (Mikolov et al., 2013) on the utterances of the training set, and was frozen during training.
        *    Vocabulary size was around 21K, and out-of-vocabulary words were mapped to a special token [UNK], randomly initialized.
        *    trained with the Adam optimizer. Early stopping was used on the validation set with a patience of 5 epochs and a maximum number of epochs of 100
        *    在validation set上表現最佳的epoch通常在錢10個epoch，batch size為1，也就是一個training step只取一個conversation，有嘗試過Batch sizes of 1, 2, 4, 8, 16，沒有顯著的效果差異

## Quantitative results(定量)

*    Performance comparison
        * 因為CRF extra input 加入SC(speaker-change)讓performance提升，且10次acc之標準差變小。Vanilla CRF是base model採用的CRF架構，Softmax指的是BiLSTM完直接套FC layer+softmax而沒有用CRF
            ![](https://i.imgur.com/f6uaSEC.png)
        * Note that our confusion matrices were row-wise normalized by class size. So we use the terms accuracy (per class) to denote diagonal values (equivalent to recall or hit rate), and miss rate for off-diagonal values.
        * 缺點
            * label "sd" 是數量最多的DA label，但是該label之acc表現不如base model。
            * In addition, the performance drop regarding sd can be interpreted as a consequence of the trade-off between sd and sv, since the distinction between them was very hard to make even by annotators
            ![](https://i.imgur.com/a66cJ49.png)

*    The benefits of considering speaker information vary across DA labels
    有些utt判斷其DA不需要上下文，所以我們的model在判斷Non-verbal (x), Conventional-closing (fc), Appreciation(ba), and Yes-No-Question (qy)這四類DA時與base model相比效果差不多。而我們的model對那些需要speaker-change這項information的DA label就會有明顯進步
*    Ensembling and joint training
        * First, an ensembling approach that combines the predictions of the two trained models by averaging their emission and transition scores (respectively).
        * Second, a joint training approach that combines the two models into a new one and trains it from scratch. And transition scores are computed as:
        ![](https://i.imgur.com/MVtr3yv.png)
        第二種作法結果不好，推測原因是G_basis blurred the label transition patterns
*    Ablation studies
        *    為了證明我們採用modified CRF是最好的選擇，我們訓練了另外兩個base model，都是採用Vanilla CRF，並分別concatenate SI與SC在u^t後面(分別是Table2的b1與b2)，確實效果都比b還要好，但是與a還是有大落差。
        *    而為了實驗的完整性，我們重複上述實驗在我們的model，但是效果卻沒有比較好，所以可歸納出speaker information在BiLSTM與CRF各被採用到並不會比較好，該information被CRF用到就夠了
        *    從Table2可看出，SC比SI有用

##  Qualitative results
![](https://i.imgur.com/h0l9jgr.png)
* 語者沒變時的matrix通常會跟錢一時間點的DA一致
* 最右邊的matrix類似於(左邊加中間的matrix)

## Discussion
*    作者嘗試過用bidirectional LSTM(also with last pooling) & bidirectional LSTM with self-attention mechanism當作utterance encoder，但是效果沒有比較好
*    推測self-attention效果沒有RNN好，**SwDA dataset的utterance之token數太少**(68.7%的utterance都沒有超過10個token)，
*    反觀RNN:On such short sequences, a RNN with a 300-dimensional hidden layer is very likely able to keep the full sequence into memory
*    As far as why a **forward RNN** suffices, it should be noted that with **last pooling**, the last time step corresponds to the first annotation of the backward RNN. **This is not adding much information to the last annotation of the forward RNN, which represents the entire sequence.**

*    Future research should be devoted to address the limitation of the Markov property of CRF layer, by developing a model that is capable of capturing longer-range dependencies within and among the three sequences: that of speakers, utterances, and DA labels.