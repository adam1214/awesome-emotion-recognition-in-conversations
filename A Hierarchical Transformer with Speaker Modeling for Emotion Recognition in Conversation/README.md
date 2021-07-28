### [Paper: A Hierarchical Transformer with Speaker Modeling for Emotion Recognition in Conversation](https://arxiv.org/pdf/2012.14781.pdf)
## Abstract
*    ERC can be regarded as a personalized and interactive emotion recognition, which is supposed to consider not only the semantic information of text but also the influences from speakers.
*    以往的方式多對每一對(2人)speakers建model，但這樣運算成本太高，model難以被擴展，且只能考慮到local context，為了解決此問題，我們簡化model變成 binary version: Intra-Speaker and Inter-Speaker dependencies, without identifying every unique speaker for the targeted speaker
*    To better achieve the simplified interaction modeling of speakers in Transformer, which shows excellent ability to settle(處理、解決) long-distance dependency, we design three types of masks and respectively utilize them in three independent Transformer blocks
*    3 designed masks respectively model : 
        *    conventional context modeling
        *    Intra-Speaker dependency
        *    Inter-Speaker dependency
        *    utilize the attention mechanism to automatically weight them
## Introduction
*    ERC is a task to predict the emotion of the current utterance expressed by a specific speaker according to the context (Poria et al. 2019b), which is more challenging than the conventional emotion recognition only considering semantic information of an independent utterance.
*    To precisely predict the emotion of a targeted utterance, both the semantic information of the utterance and the information provided by utterances in the context are critical
*    We denote this kind of information with modeling speakers’ interactions as speaker-aware contextual information
*    Fig. 1(a), Self and InterSpeaker dependencies establish a specific relation between every two speakers and construct a fully connected relational graph (DialogueGCN (Ghosal et al. 2019))
*    Although DialogueGCN can achieve excellent performance with Self and Inter-Speaker dependencies, this speaker modeling is easy to be complicated with the number of speakers increasing (Fig. 1(b)). This limitation leads to DialogueGCN only considering the local context in a conversation.
![](https://i.imgur.com/duBcZvO.png)
*    Therefore, it is appealing to introduce a simple and general speaker modeling, which is easy to extend in all scenes and realize in other models so that long-distance context can be available. (TRansforMer with Speaker Modeling (TRMSM))
        *    simplify the Self and Inter-speaker dependencies to a binary version (Fig. 1(a), 左半邊)
            *    Intra-Speaker dependency
            *     Inter-Speaker dependency
        *    settle long-distance dependency by two levels hierarchical Transformer
            *    sentence level (BERT encodes the semantic representation for a targeted utt)
            *    dialogue level (Transformer is used to capture the information from contextual utterances)
        *    dialogue-level Transformer with three masks (three independent Transformer blocks)
            *    Conventional Mask for conventional context modeling
            *    Intra-Speaker Mask for Intra-Speaker dependency
            *    Inter-Speaker Mask for Inter-Speaker dependency
            *    三個block各有不同的預測結果, utilize the attention mechanism to automatically weight and fuse them
            *    Besides, we also apply two other simple fusing methods: Add and Concatenation to demonstrate the advancement of the attention
*    Using Datasets:IEMOCAP & MELD

## Methodology
![](https://i.imgur.com/q3GmuMX.png)
### Sentence-Level Encoder
*    採用BERT (Encoder of Transformer)
    *    訓練BERT不需要有label資料，只要採用一堆文本資訊即可，因為其目的是用來做word embeding
*    因為BERT的encoder layers維度是768，受限於此，我們每次只能餵入utt內的words給BERT(solely used to encode the sentence-level context in a single utt)而不是多個utt同時餵入來考慮global contextual information
![](https://i.imgur.com/MaFWMEj.png)
*    接著將utt內的每一個word之embeding結果過maxpooling得該utt的representation
![](https://i.imgur.com/Dm2A6pg.png)
*    最後整段Conversation C:![](https://i.imgur.com/A73R544.png)(d_u是每一utt的embeding dim., N指的是有N個utts)
### Dialogue-Level Encoder
*    the same structures of this three Transformer blocks
*    to avoid the absence of positional information in C, an Absolute Positional Embedding is added to every representation in C:
![](https://i.imgur.com/PaLyRQt.png)
        *    PE(0:N)的dim.與C相同
*    self-attention中的matrix Q, K, V由C做無bia之線性投影得到，matrix大小都是N*d_a
*    Self-attention is calculated by:
![](https://i.imgur.com/Gv8y6CQ.png)
        *    上式*代表element-wise multiplication
        *    M(N*N) is the utilized mask which is a square matrix whose noninfinite elements equal 1
*    採用Multi-Head Attention, MHA來得到不同面向的資訊。 And then the outputs of all heads are concatenated and projected to O with the same size of C
*    After the Attention module, Position-wise FeedForward Network (FFN) module is deployed to produce output F (兩層fully-connected layer)
*    MHA and FFN are both residually connected (Add & Norm的Add)
*    補充 LayerNorm & BatchNorm
        *    BatchNorm是對一個batch內的data之每一個dim.的值取mean=0, var=1
        *    LayerNorm是對每一筆data的所有dim.的值取mean=0, var=1, 常用於RNN，而transformer類似於RNN，所以LayerNorm也多用在transformer

![](https://i.imgur.com/Xq37AOJ.png)
(7)式最後的output為N*d_u dim.，會有三個(7)式output產生，分別由三個transformer blocks產出
*    Conventional Mask:sets all the elements of itself to 1, which means that every targeted utterance can get access to all the contextual utterances
*    Intra-Speaker Mask:only considers those contextual utts which are the speaker tag of the targeted utt(自身語者的utt都設成1，其他設成-inf)
*    Inter-Speaker Mask:與Intra-Speaker Mask相反，自身語者的utt都設成-inf，其他設成1
### Fusing Method
*    3個transformer blocks各產出一個N*d_u matrix，要將他們得到的information作融合再去做分類，三種方式:
        *   Add (R的dim:N*d_u, Fig 2(i))
            ![](https://i.imgur.com/qoYfIgC.png)
        *   Concatenation: Different from Add operation, Concatenation can implicitly choose the information which is important for the final prediction due to the following linear projection of classifier (R的dim:N*3d_u, Fig 2(ii))
            ![](https://i.imgur.com/xv2IkX1.png)
        *    Attention:As the contributions of different speaker parties are diversely weighted, it is feasible that the model automatically chooses the more important information. Therefore, we utilize the widely used attention to achieve this goal.
            ![](https://i.imgur.com/ckOblwm.png)
            
        Oi dim:(3,du)(i指的是dialog編號), wF(trainable parameter) dim.:(1,du), alpha dim:(1,3), Ri dim:(1,du), 最後R dim:(N,du)

### Classifier
*    dialogue-level output is fed to a classifier which predicts the final emotion distributions
![](https://i.imgur.com/gig77wX.png)
*  Classifier is trained by a cross-entropy loss function 
![](https://i.imgur.com/D2Aof19.png)
## Experimental Setup
### Datasets
*    IEMOCAP & MELD are multi-modal datasets that contain three modalities(影像、音檔、逐字稿)，此篇paper只考慮textual modality
![](https://i.imgur.com/KIhTp3a.png)
*    IEMOCAP:10個spk，只考慮其中6類emo(ang, hap, neu, sad, exc, fru)，ses01-04當training set, ses05當testing set，而training set又切出20%當validation set
*    MELD:超過1400位spk.conversations collected from the TV series Friends.7類情緒:neu, joy, surprise, ang, sad, fear, disgust

### Compared Methods
![](https://i.imgur.com/vvNg8U1.png)
![](https://i.imgur.com/To6L6rX.png)
*    TRM:our model without Intra-Speaker Blocks and Inter-Speaker Blocks

### Implementation
*    sentence-level encoder
        *    an uncased BERT-base model is adopted
        *    BERT有兩種模型:BERT-Base & BERT-Large，兩種眉型都有uncased & case兩種版本
        *    Uncased 在使用WordPiece分詞之前都轉換為小寫格式，並剔除所有Accent Marker(口音標記)，而Cased 會保留它們。項目作者表示一般使用Uncased模型就可以了，除非大小寫對於任務很重要才會使用Cased版本
*    dialogue-level encoder
        *    the dimension of dialogue-level representation(d_u) is set to 300 for IEMOCAP and 200 for MELD
        *    the number of transformer layers is set to 6 for IEMOCAP and 1 for MELD
        *    the number of heads is set 6 for IEMOCAP and 4 for MELD
        *    dropout rate is set to 0.1
        *    models are trained using AdamW
        *    batch size = 1( Due to the parallel prediction of utterances in one conversation)

## Results and Discussions
### Overall Results
*    For IEMOCAP, weighted-F1 (wF1) score is used as the metric
*    However, the data proportion of MELD is in a severely imbalanced condition. Therefore, the weighted-F1 score is not that proper and enough for MELD. To balance the contributions of large classes and small classes, we follow Zhang et al. (2020) and also use the average value of macro F1 score and micro F1 score as one metric(相加除以2得mF1)
*    Table2中單純BERT的效果很差(This result may indicate that IEMOCAP contains considerable utterances that cannot be predicted only depending on the semantic information, which is out of the conversational context.)
*    For MELD, as shown in Tab. 3, BERT outperforms other state-of-the-art models by a great margin, which indicates the importance of external knowledge brought by BERT
*    在MELD，採用CNN得到55.02 wF1並且在scLSTM得到55.9 wF1，證明MELD的上下文對model所能提供的資訊有限，但是TRM系列的model用到這樣有限資訊的上下文還是能夠比BERT更好
*    TRMSM-Att and TRMSM-Cat outperform TRMSM-Add, which indicates the importance of different aspects of speaker information requiring to be treated differently
*    TRMSM-Att outperforming TRMSM-Cat demonstrates that automatically and explicitly picking up speaker-related information is better than the implicit way.

### Model Analysis
#### Ablation Study
*    To better understand the influences of masks on our models, we report the results of the models removing the Transformer blocks with different masks on IEMOCAP and MELD
![](https://i.imgur.com/VumPrJs.png)
SM指的是Intra-Speaker & Inter-Speaker Masks, CM指的是Convention
Mask
*    We attribute it to that speaker modeling does not drop the contextual information from conversations, and on the contrary, speaker modeling can guide the model to extract more effective information to the final prediction

#### Effect of Range of Context
*    To find out the influence of the range of context on our model, we train TRMSM-Att with different ranges of available context on IEMOCAP
*    We utilize different windows (−x, y) to limit the context, where x, y is respectively the number of utterances in prior context and post context.
*    As illustrated in Fig. 4, with the window widened, the performance increases as shown in both model
![](https://i.imgur.com/Tzjrbyv.png)

#### Effect of Number of Layers
![](https://i.imgur.com/ojCoXqZ.png)
*    As the number of layers increasing, the F1 scores of emotions in IEMOCAP normally expand.
*    in MELD, increasing the number of layers gradually hurts the performance to be 0 of F1 on emotions Fear and Disgust which are classes with the fewest data. We think the reason may be that MELD suffers from data imbalance and increasing the number of layers leads to severer overfitting on small classes

### Case Study
![](https://i.imgur.com/0GmI42H.png)
*    上圖的TRMSM沒有Conventional Blocks，因為這樣曾能純考慮剩下兩個spk blocks的attention from fusing比例
*    Attention score:We average the attention scores of all self-attention heads in the top layer of Transformer(個人見解:因為是multi-head，每個head先將(QK*M)/d_a^(1/2)取softmax後，再取平均得到此attention score，attention score有0是因為那些位置被mask了(-inf)，所以那些位置的值會是負無窮大，自然softmax值是0
