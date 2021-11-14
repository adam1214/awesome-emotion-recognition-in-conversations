### [Paper: FULLY SUPERVISED SPEAKER DIARIZATION](https://arxiv.org/pdf/1810.04719.pdf)
### [Source Code](https://github.com/google/uis-rnn)

## Abstract
*    propose a fully supervised speaker diarization approach, named unbounded interleaved-state recurrent neural networks (UIS-RNN)
*    Given extracted speaker-discriminative embeddings (a.k.a. d-vectors) from input utterances, each individual speaker is modeled by a parameter-sharing RNN, while the RNN states for different speakers interleave in the time domain
*    This RNN is naturally integrated with a distance-dependent Chinese restaurant process (ddCRP) to accommodate an unknown number of speakers

## INTRODUCTION
*    Aiming to solve the problem of “who spoke when”, most existing speaker diarization systems consist of multiple relatively independent components
        1.   A speech segmentation module, which removes the non-speech parts, and divides the input utterance into small segments
        2.   An embedding extraction module, where speaker-discriminative embeddings such as speaker factors, i-vectors, or d-vectors are extracted from the small segments
                *    recent work has shown that the diarization performance can be significantly improved by replacing i-vectors with neural network embeddings, a.k.a. d-vectors 
        3.    A clustering module, which determines the number of speakers, and assigns speaker identities to each segment
                *    unsupervised, 如Gaussian mixture models, k-means...，缺點是無法用label data去提升該module的performance
        4.    A resegmentation module, which further refines the diarization results by enforcing additional constraints
*    In this paper, we replace the unsupervised clustering module by an online generative process that naturally incorporates labelled data for training. We call this method unbounded interleaved-state recurrent neural network (UIS-RNN)
        *    Each speaker is modeled by an instance of RNN, and these instances share the same parameters
        *    An unbounded number of RNN instances can be generated
        *    The states of different RNN instances, corresponding to different speakers, are interleaved in the time domain.
        *    Within a fully supervised framework, our method in addition handles complexities in speaker diarization: it automatically learns the number of speakers within each utterance via a Bayesian non-parametric process, and it carries information through time via the RNN. 

##  BASELINE SYSTEM USING CLUSTERING
*    Our diarization system is built on top of the recent work by Wang et al
        *    component 1跟2都沒有變，並將component 3從非監督式的聚類法改成unbounded interleaved-state RNN
        *    關於 component 1, A speech segmentation module並移除non-speech parts: A simple voice activity detector (VAD) with only two full-covariance Gaussians is used to remove non-speech parts, and partition the utterance into nonoverlapping segments with max length of 400ms.
        *    關於 component 2, embedding extraction module:a text-independent speaker recognition network is used to extract embeddings from sliding windows of size 240ms and 50% overlap
        *    Then we average window-level embeddings to segment-level d-vectors, and feed them into the clustering algorithm to produce final diarization results
        *    The workflow of this baseline system is shown in Fig. 1.
        ![](https://i.imgur.com/tdyYKuN.png)
        *    The text-independent speaker recognition network for computing embeddings has three LSTM layers and one linear layer. The network is trained with the state-of-the-art generalized end-to-end loss.此篇作者有retrain該model
        
## UNBOUNDED INTERLEAVED-STATE RNN
###  Overview of approach
*    Given an utterance, from the embedding extraction module, we get an observation sequence of embeddings X = (x_1, x_2, ..., x_T), 每一個小x都是一個d-vector，對應於該utt的其中一個segment
*    In the supervised speaker diarization scenario, we also have the ground truth speaker labels for each segment Y = (y_1, y_2, ..., y_T)

![](https://i.imgur.com/dWBXEAL.png)

*    跟松霖DED一樣的概念
*    sequence generation對應DED取cross entropy loss
*    speaker assignment對應DED的emotion assignment(中餐廳流程)
*    speaker change對應DED的emotion shift
### Details on model components

#### Speaker change
*    speaker change的probability可以用RNN導出，但是這邊跟松霖DED一樣都是採用一個常數對:p_0 & (1-p_0)

#### Speaker assignment process
*    One of the biggest challenges in speaker diarization is to determine the total number of speakers for each utterance
*    To model the speaker turn behavior in an utterance, we use a distance dependent Chinese restaurant process (ddCRP), a Bayesian nonparametric model that can potentially model an unbounded number of speakers
*    當z_t = 0，speaker沒變
*    當z_t = 1:
![](https://i.imgur.com/31WOeld.png)
*    K_(t-1)是從第1秒到第(t-1)秒的unique speakers總數
*    N_(k,(t-1))是從第1秒到第(t-1)秒speaker k的block數量，一個block定義為k語者連續說話的最長連續segment
*    alpha是到新語者的prob.之分母
*    The joint distribution of Y given Z 
![](https://i.imgur.com/Va0cXii.png)
跟松霖DED不一樣的地方是分母的部分不是加總所有speaker的block數，而是要去除掉(t-1)時間點的speaker。
*    為甚麼是alpha的(K_T-1)次方?
*    分母的"1"，是指z_t=1發生的次數嗎?

#### Sequence generation
*    Our basic assumption is that, the observation sequence of speaker embeddings X is generated by distributions that are parameterized by the output of an RNN
*    This RNN has multiple instantiations, corresponding to different speakers, and they share the same set of RNN parameters θ
*    we use gated recurrent unit (GRU) as our RNN model, to memorize long-term dependencies.
![](https://i.imgur.com/qLeQrXR.png)
![](https://i.imgur.com/Bc9pks2.png)

#### Summary of the model
*    ![](https://i.imgur.com/RbaetHu.png)
*    Z and λ are omitted for a simple demonstration
*    current stage是實線，y_7有四種選項:1, 2, 3(existing speakers), and 4 (a new speaker)
*    產出新的observation x_7的機率(虛線)需依賴y_[6](previous label assignment sequence)與x_[6]的資訊
        *    若要predict新label，input就是x_0與h_0
        *    若要predict歷史紀錄已存在的label，假設當前時間是t，且比當前時間小並且最接近當前時間的相同predict label的時間是m，input就是x_m與h_m
*    h要如同式10一樣去連結(虛線)小於並最接近當前時間的相同label，而m是GRU_h的所有層的output，其連線方式也是如此

### MLE Estimation
*    ![](https://i.imgur.com/O23Oxgf.png)

*    speaker unchanged probability算法(lambda)
![](https://i.imgur.com/zGG74hD.png)
分母是指seagment之間格數，所以要減去N
*    theta(GRU的參數) & sigma^2(某個stage的GRU之所有層數的output之變異數)的參數更新方式一樣(SGD，一次取b個utt):
![](https://i.imgur.com/0wQDmCc.png)

*    alpha(ddCRP)之參數更新方式(SGD，一次取b個utt)(ln後面那一項就是式(8)):
![](https://i.imgur.com/1Qn9djR.png)

*    上述的參數更新learning rate皆固定

### MAP Decoding
![](https://i.imgur.com/LDFm9Rp.png)
*    透過此演算法，將O(T!)壓到O(T^2)，並且觀察到一段utterance中通常只會有常數C個人做對話，進而將時間複雜度壓到O(T) (採用beam search)

## EXPERIMENTS
### Speaker recognition model(用來output d-vector)
*    加入更多訓練資料
*    原本的model是用1600ms的window size做訓練，與此篇採用的240ms來infernece d-1 vector不一致，故改採用[240ms, 1600ms]的uniform distribution window size重新訓練
*    V1 V2 V3差別

![](https://i.imgur.com/E3jDa92.png)
EER: The speaker verification Equal Error Rate

### UIS-RNN setup
*    sequence generation model是採用一層512個GRU cells 搭配tanh activation，緊接著兩層512個neurons的DNN都是搭配ReLU activation，此兩層DNN就是對應式9的m(GRU output)
*    Decoding過程採用beam size=10

### Evaluation protocols
*    採用pyannote.metrics library做evaluation
*    We evaluate on single channel audio.
*    We exclude overlapped speech from evaluation.
*    We tolerate errors less than 250ms in segment boundaries.
*    We report the confusion error, which is usually directly referred to as Diarization Error Rate (DER) in the literature

### Datasets
*    For the evaluation, we use 2000 NIST Speaker Recognition Evaluation (LDC2001S97), Disk-8, which is usually directly referred to as “CALLHOME” in literature
        *    It contains 500 utterances distributed across six languages: Arabic, English, German, Japanese, Mandarin, and Spanish. Each utterance contains 2 to 7 speakers
        *    Since our approach is supervised, we perform a 5-fold cross validation on this dataset. We randomly partition the dataset into five subsets, and each time leave one subset for evaluation, and train UISRNN on the other four subsets. Then we combine the evaluation on five subsets and report the averaged DER.
        *    we also tried to use two off-domain datasets for training UIS-RNN: (1) 2000 NIST Speaker Recognition Evaluation, Disk6, which is often referred to as “Switchboard”; (2) ICSI Meeting Corpus. We first tried to train UIS-RNN purely on off-domain datasets, and evaluate on CALLHOME; we then tried to add the off-domain datasets to the training partition of each of the 5-fold.

### Results
*    ![](https://i.imgur.com/YlaPDWS.png)
*    k-means與spectral都是非監督式學習的聚類方式
*    Disk-6與ICSI是off-domain training dataset
*    From the table, we see that the biggest improvement in DER actually comes from upgrading the speaker recognition model from V2 to V3. This is because in V3, we have the window size consistent between training time and diarization inference time, which was a big issue in V1 and V2.

## CONCLUSIONS
*    One interesting future work direction is to directly use accoustic features instead of pre-trained embeddings as the observation sequence for UIS-RNN, such that the entire speaker diarization system becomes an end-to-end model.
