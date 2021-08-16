### [Source Code](https://github.com/30stomercury/Interaction-Aware-Attention-Network)

## ABSTRACT
*    Previous research has largely focused on performing SER by modeling each utterance of the dialog in isolation without considering the transactional and dependent nature of the human-human conversation
*    In this work, we propose an interaction-aware attention network (IAAN) that incorporate contextual information in the learned vocal representation through a novel attention mechanism.

## INTRODUCTION
*    to obtain better characterize a target speaker’s current emotion state, his/her own previous state and behaviors from his/her interacting partners are two prime contributions in this transactional aspect of emotion.
*    our aim is to further improve the speech emotion recognition in spoken dialogs by learning to embed these transactional aspect into vocal representation using attention network
*    許多model有學習到上下文information，但是:
        *    the emotionally-relevant information embedded in the current utterance as a result of the transactional, i.e., transitional and contagious, effect is not explicitly learned and integrated in the representation of the current utterance. (專屬情緒方面的資訊沒有特別被embedded)
*    To address this issue, we propose a complete architecture of interaction-aware attention network (IAAN), which is built based on attention-based gated recurrent units (GRUs)
*    By including two contextual utterances as a unit of transactional frame, i.e., the previous utterance of the current speaker, and the previous utterance of the interlocutor, we devise an attention mechanism that embed the transactional information into the current utterance.

## RESEARCH METHODOLOGY

### Dataset Description
*    IEMOCAP: 只採用4類(ang, hap, neu, sad)
        *    happiness and excitement are considered together as happiness

### Acoustic Low-level Descriptors
*    We extract acoustic low-level descriptors (LLDs) based on Emobase 2010 Config using the openSMILE toolkit , including features such as Mel-Frequency Cepstral Coefficients(MFCCs), pitch and their statistics in each short frame of an utterance
        *    obtain a sequence of a total of 45 dimensional frame-level acoustic features for each utterance
        *    apply speaker-dependent z-normalization for each descriptor, and we further downsample the frame numbers by averaging feature values every five frames to reduce the computational cost.

### Interaction-aware Attention Network (IAAN)
![](https://i.imgur.com/PWrXvH1.png)

*    integrate influences of the contextual information between interlocutors within a transactional frame to perform emotion recognition
#### Transactional Context
*    一筆data: x:(U_c, U_p, U_r) y:(U_c的情緒label)
        *    U_c:當前utt
        *    U_p:與U_c同語者的前一個utt (情緒label沒有用到)
        *    U_r:與U_c不同語者的前一個utt (情緒label沒有用到)

####  Interaction-aware Attention Representation
*    對於each frame of transactional context，將U_p與U_r encode成fixed-length utt-leavel features h_p and h_r (用GRU with Bahdanau attention mechanism來encode)
*    對於U_c，先採用bidirectional GRU做encode (下圖中的frame指的是每一utt用openSMILE取feature時，會有好幾個45維的frame產生)
![](https://i.imgur.com/beTdKZu.png)
        *    這邊沒有跟著採用Bahdanau attention，而是採用interaction-aware attention
        *    The interaction-aware attention is designed to capture the affective transition (previous utterance of the same target speaker) and affective influence (previous utterance of the interlocutor) into the representation of current target utterance
        *    Hence, while encoding the current utterance’s attentive representation h_c, the previous utterance information in the h_p and h_r are integrated into current utterance encoder.
        *    定義alpha(attention weight)
![](https://i.imgur.com/BW2z2vd.png)
        *    再用alpha做加權總合得到current utterance representation
![](https://i.imgur.com/pzaLba7.png)

#### Emotion Classification Network
*    within every transactional frame, 先concat h_c, h_p, h_r，過一層全連接層再過relu，再過一層全連接層，然後softmax:
![](https://i.imgur.com/2CcG5Rn.png)
![](https://i.imgur.com/ZsYwPfo.png)
*    整個架構採cross-entropy loss

## EXPERIMENTAL SETUP AND RESULTS

### Experimental setup
*    the hidden unit dimension is set to 512 for two GRUs
*    256 for each direction of BiGRU
*    lr = 0.0001
*    mini-batch size is set as 64
*    apply 90% dropout to each GRU and BiGRU cells as well as the output of first projection layer
*    add a weight decay of 0.001 to all weights and biases in the projection layers
*    Adam optimizer
*    carry out early stopping by observing the performance on validation set in every 100 training epochs.
*    evaluate the performance
        *    UA (unweighted acc)
        *    WA (weighted acc)
*    5-fold leave-one-session-out (LOSO) cross validation.

#### Baseline Methods 
![](https://i.imgur.com/W7Cl9FL.png)
*    BiGRU+ATT:A BiGRU network with the classical attention (ATT) trained using current utterances only.
*    BiGRU+IAA: The framework and inputs are same as IAAN, but instead of the joint concatenated representation, the predictions only depend on current utterance’s representation.
*    RandIAAN: The IAAN approach but trained using the randomly selected auxiliary utterances in the dialog as a transactional frame.

### Result and Analysis

####  Analysis
![](https://i.imgur.com/noTnWIv.png)
![](https://i.imgur.com/fPloRSm.png)
*    In Case 1 and Case 2, we observe that once U_c shares the same emotion (even partially) as their immediate preceding emotional contexts, IAAN achieves the best recognition rates.
*    the condition where the previous utterances have completely different emotion from U_c (Case 3) results in lowest accuracies.
*    More interesting, if we examine the type of emotions of U_c in Case 3, the emotions are dominated by neutrality that accounts for 37% of data, where angry, happiness, sadness are 24.8%, 22.2% and 14.9%, respectively
*    Furthermore, the UA of neutral category is only 47.6% in Case 3, which suggests that the emotional characteristics of neutrality has less relevance from their emotional contexts as compared to others (neutrality seems to be more context-free)
*    IAAN對neu label的predict錯誤率比較高，因為neu比較不需要抓取上下文訊息(不受上下文影響)

