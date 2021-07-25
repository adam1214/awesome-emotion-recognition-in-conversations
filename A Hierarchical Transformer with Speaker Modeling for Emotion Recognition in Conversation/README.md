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
*    to avoid the absence of positional information in C, an Absolute Positional Embedding is added to every representation in C:![](https://i.imgur.com/PaLyRQt.png)
        *    PE(0:N)的dim.與C相同
*    self-attention中的matrix Q, K, V由C做無bia之線性投影得到，matrix大小都是N*d_a
*    Self-attention is calculated by:![](https://i.imgur.com/Gv8y6CQ.png)
        *    上式*代表element-wise multiplication
        *    M(N*N) is the utilized mask which is a square matrix whose noninfinite elements equal 1
*    採用Multi-Head Attention, MHA來得到不同面向的資訊。 And then the outputs of all heads are concatenated and projected to O with the same size of C
*    After the Attention module, Position-wise FeedForward Network (FFN) module is deployed to produce output F (兩層layer)
*    MHA and FFN are both residually connected
![](https://i.imgur.com/Xq37AOJ.png)
(7)式最後的output式N*d_u dim.，會有三個(7)式output產生，分別由三個transformer blocks產出
# 複習LayerNorm
# 查Position-wise FeedForward Network