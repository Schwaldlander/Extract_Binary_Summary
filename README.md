# Extract_Binary_Summary
Extractive Summerization of business negociations
# Model Description
It is more sensible to consider context embedding and attention mechanism
to draw inference to this task of multi-speaker conversation summarization.
## Sequential Text Analysis
Let’s start from basics, and let the model learn the ranking of utterances without graphic attributes.
We divide training and validation set with a ratio of 4:1. We attempt XGBoost, LGBM, Transformer
and BiLSTM models, and we found that BiLSTM performs the best in classification. This is explicable,
since BiLSTM can handle relations of textual input before and after a current position.
Evidently, the recall is lower than accuracy, which means there are important utterances being
falsely identified as trivial. This is where graph structure comes to play to bring into attention correlations between utterances, be it general or local. For tunning, we train and validate with different
super-parameters including hidden dimmension, number of layers.
## Graph Representation
Since connections are not affine nor complete, as suggested by Zhouxing Shi et al., we need multilayer
perceptrons to represent non linear relationships between nodes, where lower layers extract learn simple
patterns, and as the information propagates through the network, higher layers learn more abstract and
complex representations. To stablize results and avoid gradient vanishing, ReLU activation function is
adopted. Currently we set MLP layer as two layer with its output channel set as 128. Since dialogue
length ranges from 128 to 980, it might be tempting to regulate each batch input into the same
utterance length, yet this would risk breaking graphic information and is not helpful.
## Encoder Decoder Structure
GRU is applied to provide embedding for concatenated information of both MLP-processed graph relation representations and encoded dialogue text, due to its fast computation and capabilities of handling
shorter dependencies. Following the model proposed by Jiaao Chen et al., we apply graph attention convolution layer embed utterance connections. The following equation shows how we incorporate
utterance features with graph topology represented.

