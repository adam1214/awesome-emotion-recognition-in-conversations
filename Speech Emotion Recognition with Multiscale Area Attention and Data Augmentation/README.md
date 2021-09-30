### [Paper: SPEECH EMOTION RECOGNITION WITH MULTISCALE AREA ATTENTION AND DATA AUGMENTATION](https://arxiv.org/pdf/2102.01813v1.pdf) (Accepted by ICASSP 2021)
### [Source Code](https://github.com/makcedward/nlpaug)

## ABSTRACT
*    In SER, emotional characteristics often appear in diverse forms of energy patterns in spectrograms
*    Typical attention neural network classifiers of SER are usually optimized on a fixed attention granularity(粒度)
*    In this paper, we apply multiscale area attention in a deep convolutional neural network to attend emotional characteristics with varied granularities and therefore the classifier can benefit from an ensemble of attentions with different scales.
*    To deal with data sparsity, we conduct data augmentation with vocal tract length perturbation (聲道長度擾動)(VTLP) to improve the generalization capability of the classifier.

## INTRODUCTION
*    there are still deficiencies in the research of SER, such as data shortage and insufficient model accuracy
*    In SER, emotion may display distinct energy patterns in spectrograms with varied granularity of areas.
*    Therefore, in this paper, we introduce multiscale area attention to a deep convolutional neural network model based on Head Fusion to improve model accuracy
*    Furthermore, data augmentation is used to address the data scarcity issue.
*    Our main contributions are as follows:
        *     the first attempt for applying multiscale area attention to SER
        *     We performed data augmentation on the IEMOCAP dataset with vocal tract length perturbation (VTLP) and achieved an accuracy improvement of about 0.5% absolute.
## METHODOLOGY
![](https://i.imgur.com/lh1vb7J.png)

### Convolutional Neural Networks
*    First, the Librosa audio processing library [18] is used to extract the logMel spectrogram as features, which are fed into two parallel convolutional layers to extract textures from the time axis and frequency axis, respectively
*    The result is fed into four consecutive convolutional layers and generates an 80-channel representation
*    Then the attention layer attends on the representation and sends the outputs to the fully connected layer for classification
*    Batch normalization is applied after each convolutional layer.