### [Paper: NEURAL CRF TRANSDUCERS FOR SEQUENCE LABELING](https://arxiv.org/pdf/1811.01382.pdf)

## ABSTRACT
*    Various linearchain neural CRFs (NCRFs) are developed to implement the nonlinear node potentials in CRFs, but still keeping the linear-chain hidden structure
*    In this paper, we propose NCRF transducers, which consists of two RNNs, 
        *    one extracting features from observations
        *    the other capturing (theoretically infinite) long-range dependencies between labels. 
        *    Different sequence labeling methods are evaluated over POS tagging, chunking and NER (English, Dutch). Experiment results show that NCRF transducers achieve consistent improvements over linear-chain NCRFs and RNN transducers across all the four tasks, and can improve state-of-the-art results.

## INTRODUCTION
*    A recent progress is to develop Neural CRF (NCRF) models, which combines the sequence-level discriminative ability of CRFs and the representation ability of neural networks (NNs), particularly the recurrent NNs (RNNs). These models have achieved state-of-the-art results on a variety of sequence labeling tasks, and in different studies, are called conditional neural field [10], neural CRF [12], recurrent CRF [13], and LSTM-CRF [4, 5]
        *    they are all defined by using NNs (of different network architectures) to implement the non-linear node potentials in CRFs, while still keeping the linear-chain hidden structure (i.e. using a bigram table as the edge potential)
        *    we refer to these existing combinations of CRFs and NNs as linear-chain NCRFs in general
*    This represents an extension from conventional CRFs, where both node potentials and edge potentials are implemented as linear functions using discrete indicator features.
*    In this paper, we present a further extension and propose neural CRF transducers, which introduce a LSTM-RNN to implement a new edge potential so that long-range dependencies in the label sequence are captured and modeled.
        *    In contrast, linear-chain NCRFs capture only first-order interactions and neglect higher-order dependencies between labels, which can be potentially useful in real-world sequence labeling applications

*    Additionally, the recent attention-based seq2seq models [16] also use an LSTM-based decoder to exploit long-range dependences between labels. However, both RNN transducers and seq2seq models, as locally normalized models, produce position-by-position conditional distributions over output labels, and thus suffer from the label bias and exposure bias problems [11, 17, 18]. In contrast, NCRF transducers are globally normalized, which overcome these two problems. We leave more discussions about existing related studies to section 6.

## BACKGROUND
![](https://i.imgur.com/dymghII.png)
![](https://i.imgur.com/vJQzsmB.png)
*    RNN Transducers
![](https://i.imgur.com/YjCPCRx.png)

*    To ease comparison, we will also refer to the network below the CRF layer in linear-chain NCRFs as a transcription network

## NCRF TRANSDUCERS
*    In the following, we develop NCRF transducers, which combine the advantages of linear-chain NCRFs (globally normalized, using LSTM-RNNs to implenent node potentials) and of RNN transducers (capable of capturing long-range dependencies in labels), and meanwhile overcome their drawbacks, as illustrated in Table 1.

![](https://i.imgur.com/N15pibn.png)

### Model definition
![](https://i.imgur.com/BZi42WY.png)

### Neural network architectures
![](https://i.imgur.com/naNb1ho.png)
![](https://i.imgur.com/bNuE3fU.png)
![](https://i.imgur.com/yQZPmdR.png)
*    It can be seen from above that a NCRF transducer is similar to a RNN transducer. The difference is that a RNN transducer is local normalized through softmax calculations as shown in Eq. (1), while a NCRF transducer is globally normalized, locally producing (unnormalized) potential scores.


### Potential design

![](https://i.imgur.com/fP4NPyD.png)

### Decoding and training
![](https://i.imgur.com/wXFOX0g.png)



