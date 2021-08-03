### [Paper: Directed Acyclic Graph Network for Conversational Emotion Recognition](https://arxiv.org/pdf/2105.12907v1.pdf)
### [Source Code](https://github.com/shenwzh3/DAG-ERC)

## Abstract
*    put forward a novel idea of encoding the utterances with a directed acyclic graph (DAG) to better model the intrinsic structure within a conversation, and design a directed acyclic neural network, namely DAG-ERC, to implement this idea
*    In an attempt to combine the strengths of conventional graph-based neural models and recurrence-based neural models, DAG-ERC provides a more intuitive way to model the information flow between long-distance conversation background and nearby context

## Introduction
*    ERC應用
        *    opinion mining in social media
        *    building an emotional and empathetic dialog system
*    Numerous efforts have been devoted to the modeling of conversation context
        *    graph-based methods
                *    they concurrently gather information of the surrounding utterances within a certain window, while neglecting the distant utterances and the sequential information
        *    recurrence-based methods
                *    consider the distant utterances and sequential information by encoding the utterances temporally
                *    However, they tend to update the query utterance’s state with only relatively limited information from the nearest utterances, making them difficult to get a satisfying performance.
*    According to the above analysis, an intuitively better way to solve ERC is to allow the advantages of graph-based methods and recurrence-based models to complement each other
*    This can be achieved by regarding each conversation as a directed acyclic graph (DAG) (Fig 1)
*    Fig 1:each utterance in a conversation only receives information from some previous utterances and cannot propagate information backward to itself and its predecessors through any path
![](https://i.imgur.com/9j46fKA.png)
*    by the information flow from predecessors to successors through edges, DAG can gather information for a query utterance from both the neighboring utterances and the remote utterances, which acts like a combination of graph structure and recurrence structure.
*    propose a method to model the conversation context in the form of DAG
        *    rather than simply connecting each utterance with a fixed number of its surrounding utterances to build a graph, we propose a new way to build a DAG from the conversation with constraints on speaker identity and positional relations
        *    we propose a directed acyclic graph neural network for ERC, namely DAG-ERC. Unlike the traditional graph neural networks that aggregate information from the previous layer, DAG-ERC can recurrently gather information of predecessors for every utterance in a single layer, which enables the model to encode the remote context without having to stack too many layers
        *    a relation-aware feature transformation to gather information based on speaker identity
        *    a contextual information unit to enhance the information of historical context

## Methodology
*    the nodes in the DAG are the utterances in the conversation (vertax)
*    the edge represents the information propagated from utt_i to utt_j. r_ij is the relation type of the edge
*    The set of relation types of edges is {0,1}, contains two types of relation: 1 for that the two connected utterances are spoken by the same speaker, and 0 for otherwise.
*    We impose three constraints to decide when an utterance would propagate information to another, i.e., when two utterances are connected in the DAG:
        *    Direction: A previous utterance can pass message to a future utterance, but a future utterance cannot pass message backwards.
        *    Remote information:當前utt之前一個utt of the same spk，稱u_tau。u_tau之前的utt稱Remote information，是相對較不重要的information. We assume that when the speaker speaks u_tau, he has been aware of the remote information before u_tau. 所以u_tau涵蓋了remote information to current utt u_i
        *    Local information:Usually, the information of the local context is important.  We assume that every utterance u_l in between u_tau and u_i contains local information, and they will propagate the local information to u_i
*    p(u_i)指的是u_i是哪個spk說的
*    We regard u_tau as the w-th latest utt spoken by p(u_i) before u_i(距離u_i第w近的同語者utt), where w is a hyper-parameter.
*    each u_l in between u_tau and u_i, we make a directed edge from u_l to u_i
![](https://i.imgur.com/U9zZj0F.png)
![](https://i.imgur.com/Nf7G9FG.png)
(左下角 u_3要改u_2)

### Directed Acyclic Graph Neural Network
![](https://i.imgur.com/g6h8fPe.png)

#### Utterance Feature Extraction
*    DAG-ERC regards each utterance as a graph node, the feature of which can be extracted by a pretrained Transformer-based language model. 
*    Following the convention, the pre-trained language model is firstly fine-tuned on each ERC dataset, and its parameters are then frozen while training DAG-ERC
*    employ RoBERTa-Large, which has the same architecture as BERT-Large, as our feature extractor
*    for each utt, we prepend(前置) a special token [CLS] to its tokens. Then, we use the [CLS]'s pooled embedding at the last layer as the feature representation of this utt

#### GNN, RNN and DAGNN
*    這邊的node指的是RNN裡面處理每一個utterance的單位, H指的是hidden layer
*    For each node at each layer, graph-based models (GNN) aggregate the information of its neighboring nodes at the previous layer as follows:
![](https://i.imgur.com/zOhTKsF.png)
(f(。) is the information processing function, Aggregate(。) is the information aggregation function to gather information from neighboring nodes, N_i denotes the neighbours of the i_th node)
(當前layer的當前node i是透過前一layer的鄰近node i的聚合體與前一layer的node i計算而來)
*    Recurrence-based models (RNN) allow information to propagate temporally at the same layer, while the i-th node only receives information from the (i−1)-th node:
![](https://i.imgur.com/zBNCQUH.png)
(當前layer的當前node i是透過當前layer node_(i-1)與前一layer的node i計算而來)
*    Directed acyclic graph models (DAGNN) work like a combination of GNN and RNN.  They aggregate information for each node in temporal order, and allow all nodes to gather information from neighbors and update their states at the same layer:
![](https://i.imgur.com/5Z2YfjH.png)
(當前layer的當前node i是透過當前layer的鄰近node之聚合體與前一layer的node i計算而來)
*    By allowing information to propagate temporally at the same layer, DAGNN can get access to distant utterances and model the information flow throughout the whole conversation, which is hardly possible for GNN
*    DAGNN gathers information from several neighboring utterances, which sounds more appealing than RNN as the latter only receives information from the (i−1)-th utterance.

#### DAG-ERC Layers
*    At each layer l of DAG-ERC, due to the temporal information flow, the hidden state of utterances should be computed recurrently from the first utterance to the last one.

*    For each utterance u_i, the attention weights between u_i and its predecessors are calculated by:
![](https://i.imgur.com/Mfy9Ol5.png)
(W是神經網路參數)
*    The information aggregation operation in DAG-ERC: Instead of merely gathering information according to the attention weights, we apply a relation-aware feature transformation to make full use of the relational type of edges:
![](https://i.imgur.com/pSXyFX4.png)
(這就是Aggregate(。)算法，包含兩種權重，其一是utt與utt之間的權重(alpha)，另一是relation-aware transforamtion權重
*    原本DAGNN的做法:
![](https://i.imgur.com/LHhInar.png)

(對照式(3), equation 6 is nodal information unit, the information of context M_li is only used to control the propagation of u_i's hidden state, and under this circumstance, the information of context is not fully leveraged.)
*    基於上述原因，改造DAGNN的做法，並將兩種GRU output加總:
![](https://i.imgur.com/9ztvPAJ.png)
(將式(6)GRU的兩個input顛倒餵，換句話說就是H controls the propagation of M_li)
     ![](https://i.imgur.com/sRWrFGX.png)

#### Training and Prediction
*    prediction
![](https://i.imgur.com/Q567gt3.png)
*    standard cross-entropy loss for training
![](https://i.imgur.com/T4J9qiw.png)

## Experimental Settings
### Implementation Details
*    用validation set找最佳的lr, batch size, dropout rate, DAG-ERC layer數目
*    w(往前看幾個utt of the smae spk)預設是1
*    每一utt過RoBERTa extractor得到的tensor size是1024
*    每一個DAG-ERC Layer對一個utt的output tensor size是300
*    Total 60 epochs
*    5 random runs to get avg. result

### Datasets
*    IEMOCAP取六類情緒(ang, hap, neu, sad, fru, exc), 拿Ses04的倒數20句dialog當作validation set
*    MELD取七類情緒(neutral, happiness, surprise, sadness, anger, disgust, and fear)
*    DailyDialog取七類情緒(neutral, happiness, surprise, sadness, anger, disgust, and fear)，**Since it has no speaker information, we consider utterance turns as speaker turns by default**
*    EmoryNLP取材自Friends這個電視節目秀，跟MELD一樣，但是情緒標記與取材場景與MELD不同(emotion:neutral, sad, mad, scared, powerful, peaceful, and joyful)
*    utilize only the textual modality of the above datasets for the experiments
*    evaluation metrics
        *    DailyDialog用micro-averaged F1
        *    其他用weighted-average F1
## Results and Analysis
![](https://i.imgur.com/ESJcBXv.png)
(總共4個row，第一個row屬於Recurrence-based methods, 第二個row屬於Graph-based methods, 第三個row屬於Feature extractor)
*    As shown in the table, when the feature extracting method is the same, graph-based models generally outperform recurrence-based models on IEMOCAP, DailyDialog, and EmoryNLP
*    在iemocap上(almost 70 utt per dialog，很長)，DAG-ERC又比graph-based models更優，表示DAG-ERC抓取remote information能力更好
*    然而在MELD上，graph-based models與DAG-ERC都輸給recurrence-based models. After going through the data, we find that due to the data collection method (collected from TV shows), sometimes two consecutive utterances in MELD are not coherent(連貫的). Under this circumstance, graph-based models’ advantage in encoding context is not that important.
*    換成用 RoBERTa的model有提升performance，但是還是輸DAG-ERC

### Variants of DAG Structure
幾種變形(不同的graph架構):
(1) sequence, in which utterances are connected one by one
(2) DAG with single local information, in which each
utterance only receives local information from its
nearest neighbor, and the remote information remains the same as our DAG
(3) common DAG, in which each utterance is connected with k previous utterances
![](https://i.imgur.com/WYvomGx.png)
(實驗在EmoryNLP上，因為是多人無次序對話，# Preds是average number of each utterance's predecessors)
*    Firstly, the performance of DAG-ERC drops significantly when equipped with the sequence structure
*    Secondly, our proposed DAG structure has the highest performance among the DAG structures
*    DAG with w = 2 and common DAG with k = 6, with very close numbers of predecessors, our DAG still outperforms the common DAG by a certain margin. This indicates that the constraints based on speaker identity and positional relation are effective inductive biases, and the structure of our DAG is more suitable for the ERC task than rigidly connecting each utterance with a fixed number of predecessors.
*    Finally, we find that increasing the value of ω may not contribute to the performance of our DAG, and w = 1 tends to be enough.

### Ablation Study
*    To study the impact of the modules in DAG-ERC, we evaluate DAG-ERC by removing relation-aware feature transformation, the nodal information unit, and the contextual information unit individually.
![](https://i.imgur.com/lvChKYt.png)
*    As shown in the table, removing the relationaware feature transformation causes a sharp performance drop on IEMOCAP and DailyDialog, while a slight drop on MELD and EmoryNLP. Note that there are only two speakers per dialog in IEMOCAP and DailyDialog, and there are usually more than two speakers in dialogs of MELD and EmoryNLP. Therefore, we can infer that the relation of whether two utterances have the same speaker is sufficient for two-speaker dialogs, while falls short in the multi-speaker setting.
*    拿掉nodal information unit跟拿掉contextual information unit，兩者掉的比例差不多，代表兩個因素都很重要

### Number of DAG-ERC Layers
*    the only way for GNNs to receive information from a remote utterance is to stack many GNN layers(類似CNN的Receptive field). However, it is well known that stacking too many GNN layers might cause performance degradation due to over-smoothing
*    We investigate whether the same phenomenon would happen when stacking many DAG-ERC layers
![](https://i.imgur.com/S13pc98.png)
*    從上圖看出DAG-ERC沒有因為layer數增加而造成f1值大幅度波動，所以沒有over-smoothing問題


### Error Study 

*    After going through the prediction results on the four datasets, we find that our DAG-ERC fails to distinguish between similar emotions very well, such as frustrated vs anger, happiness vs excited, scared vs mad, and joyful vs peaceful.
*    Besides, we find that DAG-ERC tends to misclassify samples of other emotions to neutral on MELD, DailyDialog and EmoryNLP due to the majority proportion of neutral samples in these datasets.
*    DAG-ERC與其他ERC model一樣，當同語者情緒有變化時的預測結果都不好(不過還是比其他model好)，如下圖
![](https://i.imgur.com/EtSbGIM.png)
