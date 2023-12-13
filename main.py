# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive/')

# Commented out IPython magic to ensure Python compatibility.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import keras.backend as K
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.metrics import precision_score, recall_score, f1_score
import collections
import random
import pandas as pd
import numpy as np

from torch_geometric.data import Data, Dataset
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.gat_conv import GATConv

import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import os
import json
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import re
import numpy as np

from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pickle
import random
random.seed(514)
from transformers import BartConfig, BartModel
from transformers.models.bart.modeling_bart import BartEncoderLayer, BartDecoderLayer
from torch_geometric.nn.norm.batch_norm import BatchNorm
import json
import argparse
from pathlib import Path
from google.colab import drive

# Flatten function
def flatten(list_of_list):
 return [item for sublist in list_of_list for item in sublist]


# Training and test set identifiers
training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']
training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
training_set.remove('IS1002a')
training_set.remove('IS1005d')
training_set.remove('TS3012c')

test_set = ['ES2003', 'ES2004', 'ES2011', 'ES2014', 'IS1008', 'IS1009', 'TS3003', 'TS3004', 'TS3006', 'TS3007']
test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in test_set])

# Load sentence transformer model
bert = SentenceTransformer('all-MiniLM-L6-v2')

###
'''
Please Feel Free to Modify these Links as your reposititory may require
'''

label_path = "/content/drive/MyDrive/DataChallenge/training_labels.json"
submission_path = Path("/content/drive/My Drive/submission4_gru_pro_1.csv")
bert_npy_link = "/content/drive/MyDrive/DataChallenge/bert_new1"
train_txt_link = "/content/drive/MyDrive/DataChallenge/training"
train_txt_pkl = "/content/drive/MyDrive/DataChallenge/train_coref_sent.pkl"
test_txt_pkl = "/content/drive/MyDrive/DataChallenge/test_coref_sent.pkl"
###

with open(label_path, "r") as file:
    training_labels = json.load(file)




def plot_losses(history_train_loss, history_val_loss):
    # Set plotting style
    #plt.style.use(('dark_background', 'bmh'))
    plt.style.use('bmh')
    plt.rc('axes', facecolor='none')
    plt.rc('figure', figsize=(16, 4))

    # Plotting loss graph
    plt.plot(history_train_loss, label='Train')
    plt.plot(history_val_loss, label='Validation')
    plt.title('Loss Graph')
    plt.legend()
    plt.show()


with open(train_txt_pkl, 'rb') as file:
    train_coref = pickle.load(file)
with open(test_txt_pkl, 'rb') as file:
    test_coref = pickle.load(file)




stoplist=["a", "an", "the",
          "of","in", "out", "on", "off", "over", "under", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
          "and", "but", "if", "or",
          "oh","uh","huh","ok","um","okay","mm",
          "no", "nor", "not",
          "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
           "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "now", "d", "ll", "m", "o", "re", "ve", "y"]#



RELATION = [
    "Continuation",
    "Explanation",
    "Elaboration",
    "Acknowledgement",
    "Comment",
    "Result",
    "Question-answer_pair",
    "Contrast",
    "Clarification_question",
    "Background",
    "Narration",
    "Alternation",
    "Conditional",
    "Q-Elab",
    "Correction",
    "Parallel",
]


def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]


TEAM_NAME = [
    "ES2002",
    "ES2005",
    "ES2006",
    "ES2007",
    "ES2008",
    "ES2009",
    "ES2010",
    "ES2012",
    "ES2013",
    "ES2015",
    "ES2016",
    "IS1000",
    "IS1001",
    "IS1002",
    "IS1003",
    "IS1004",
    "IS1005",
    "IS1006",
    "IS1007",
    "TS3005",
    "TS3008",
    "TS3009",
    "TS3010",
    "TS3011",
    "TS3012",
]
MEETING_NAME = flatten([[m_id + s_id for s_id in "abcd"] for m_id in TEAM_NAME])
MEETING_NAME.remove("IS1002a")
MEETING_NAME.remove("IS1005d")
MEETING_NAME.remove("TS3012c")

# Dictionary of English Contractions
# reference: https://www.analyticsvidhya.com/blog/2020/04/beginners-guide-exploratory-data-analysis-text-data/#1
#Add donnnu wanna gonna
contractions_dict = {
    "ain't": "are not",
    "'s": " is",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "dunno": "do not know",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "let's": "let us",
    "gonna": "going to",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "that'd": "that would",
    "that'd've": "that would have",
    "there'd": "there would",
    "there'd've": "there would have",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wanna": "want to",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what've": "what have",
    "when've": "when have",
    "where'd": "where did",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who've": "who have",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}

# Regular expression for finding contractions
contractions_re = re.compile("(%s)" % "|".join(contractions_dict.keys()))


# Function for expanding contractions
def expand_contractions(text, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions_re.sub(replace, text)


# expand_contractions example:
# df["reviews.text"] = df["reviews.text"].apply(lambda x: expand_contractions(x))
num_docs = len(MEETING_NAME)


class DataPreprocessor:
    # thanks to https://stackoverflow.com/questions/54396405/how-can-i-preprocess-nlp-text-lowercase-remove-special-characters-remove-numb
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
    def preprocess(self,sentence):
        sentence=str(sentence)
        sentence = sentence.lower()
        sentence = sentence.replace('{html}',"")
        sentence = expand_contractions(sentence)

        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', sentence)
        rem_url=re.sub(r'http\S+', '',cleantext)
        rem_num = re.sub('[0-9]+', '', rem_url)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(rem_num)

        filtered_words = [w for w in tokens if len(w) > 1 if not w in stoplist]#if len(w) > 2 if not w in stopwords.words('english')]

        stem_words=[self.stemmer.stem(w) for w in filtered_words]
        lemma_words=[self.lemmatizer.lemmatize(w) for w in stem_words]
        return " ".join(filtered_words)

data_preprocessor = DataPreprocessor()

speaker_fullname = {"PM": "product manager",
                    "ME":"marketing expert",
                    "UI":"interface designer",
                    "ID":"industrial designer"}
def resolve_first_pronouns(sentence, speaker_name):
    # Split the sentence into words
    words = sentence.split()

    # Iterate and replace pronouns
    for i in range(len(words)):
        if words[i].lower() == "we" or words[i].lower() == "our" or words[i] == "us":
            words[i] = "our team"

        elif words[i].lower() == "i" or words[i].lower() == "my" or words[i]==speaker_name:
            words[i] = speaker_fullname[speaker_name]

    # Join the words back into a sentence
    modified_sentence = ' '.join(words)
    return modified_sentence

def get_span_words(span, document):
    return " ".join(document[span[0] : span[1] + 1])

def resolve_pronouns(prediction):
  text = " ".join(prediction['document']).lower()

  document = prediction['document']
  clusters = prediction['clusters']
  span_to_rep_mention = {}
  for cluster in clusters:
      # The first span in each cluster is considered the representative mention
      rep_mention = get_span_words(cluster[0], document)
      for span in cluster:
          span_to_rep_mention[tuple(span)] = rep_mention
  sorted_spans = sorted([span for cluster in clusters for span in cluster], key=lambda x: x[0])
  # Initialize a list to hold the new tokens
  new_tokens = []
  last_end = 0
  for span in sorted_spans:
      # Get the start and end of the current span
      start, end = span
      # Append the text from the last end to the current start
      new_tokens.extend(document[last_end:start])
      # Append the representative mention for the current span
      new_tokens.append(span_to_rep_mention[tuple(span)])
      # Update the last end
      last_end = end + 1
  # Append any remaining text after the last span
  new_tokens.extend(document[last_end:])
  resolved_text = " ".join(new_tokens[2:])
  speaker = new_tokens[0]

  resolved_text = resolve_first_pronouns(resolved_text,speaker)
  return resolved_text



label_grp = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']

def select_strings(input_list):
    # Randomly shuffle the list to ensure randomness
    random.shuffle(input_list)

    # Calculate the split index for 75%
    split_idx = int(0.75 * len(input_list))

    # Split the list into selected (75%) and unselected (25%)
    selected = input_list[:split_idx]
    unselected = input_list[split_idx:]

    return selected, unselected

train_grp, val_grp = select_strings(label_grp)
def expand_grp(grp):
  t_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in grp])
  if 'IS1002a' in t_set: t_set.remove('IS1002a')
  if 'IS1005d' in t_set: t_set.remove('IS1005d')
  if 'TS3012c' in t_set: t_set.remove('TS3012c')
  return t_set
train_grp = expand_grp(train_grp)
val_grp = expand_grp(val_grp)




np.random.seed(114)

relations_dict = {'Acknowledgement': 0,
 'Alternation': 1,
 'Background': 2,
 'Clarification_question': 3,
 'Comment': 4,
 'Conditional': 5,
 'Continuation': 6,
 'Contrast': 7,
 'Correction': 8,
 'Elaboration': 9,
 'Explanation': 10,
 'Narration': 11,
 'Parallel': 12,
 'Q-Elab': 13,
 'Question-answer_pair': 14,
 'Result': 15,
 'ZEnd': 16            }

speaker_dict={"PM":0, "ME":1, "UI":2, "ID":3}

speaker_fullname = {"PM": "product manager",
                    "ME":"marketing expert",
                    "UI":"interface designer",
                    "ID":"industrial designer"}
data_preprocessor = DataPreprocessor()
class UtteranceGraphDataset(Dataset):
    def __init__(self,text_set, relations_dict = relations_dict):
        super(UtteranceGraphDataset, self).__init__(text_set)
        self.files = text_set
        self.protxt = bert_npy_link
        self.origin = train_txt_link
        self.relations_dict = relations_dict

    def len(self):
        return len(self.files)

    def get(self, idx):


        filename = self.files[idx]
        file_path = os.path.join(self.origin, filename+".txt")
        labels = []
        utterance_list = []
        speaker_list = []
        print('file',filename)

        transcription = train_coref[idx]

        npy_route = os.path.join(self.protxt, filename+'_ber.npy')
        if os.path.exists(npy_route):
            file_features = np.load(npy_route, allow_pickle=True)
        else:
            for utterance in transcription:

              prepro_utter = data_preprocessor.preprocess(resolve_pronouns(utterance))
              utterance_list.append(prepro_utter)
            file_features = np.array(bert.encode(utterance_list))
            np.save(npy_route, file_features, allow_pickle=True)

        file_features = torch.tensor(file_features)
        labels = training_labels[filename]
        with open(file_path, 'r') as file:
            relationship_types = set()
            edges = []
            edge_features = []
            features = []

            for line in file:
                source, relation, target = line.split()
                relationship_types.add(relation)
                edges.append((int(source), int(target)))
                edge_features.append(relation)

                if int(target)==len(labels)-1:

                    relationship_types.add('ZEND')
                    edges.append((int(target), int(target)))
                    edge_features.append(relation)

            # Create a one-hot encoding for relationship types



        encoded_edge_features = [self.relations_dict[rel] for rel in edge_features]

        # Convert to tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = F.one_hot(torch.tensor(encoded_edge_features, dtype=torch.long), num_classes=17)
        speak_index = []

        #features = [file_features[node_id] for node_id, _ in edges]
        features = file_features



        return {"features":features,
            "edge_index":edge_index,
            "edge_attr":edge_attr,
            "labels":labels,
            "filename":filename,
            "speaker_index":speak_index}




# data_loader = DataLoader(dataset, batch_size=1, shuffle = True)



# Randomly split into training and validation datasets
train_dataset = UtteranceGraphDataset(train_grp)
val_dataset = UtteranceGraphDataset(val_grp)
# Create DataLoaders for each set
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)







class MultiGranularityDecoder(nn.Module):
    def __init__(self, config, num_layers):
        super(MultiGranularityDecoder, self).__init__()
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(num_layers)])
        self.config = config

        self.alpha = nn.Parameter(torch.ones(1))

        # Additional cross-attention layers for discourse and action graphs
        self.discourse_cross_attention = nn.ModuleList([
            nn.MultiheadAttention(config.d_model, config.decoder_attention_heads) for _ in range(num_layers)])
        # previously input channel is d_model to handle x inputs

    def forward(self, utterance_embeddings, discourse_embeddings, memory_key_padding_mask):
        x = utterance_embeddings.unsqueeze(0)
        discourse_embeddings = discourse_embeddings.unsqueeze(0)


        for i in range(len(self.layers)):

            layer = self.layers[i]
            layer_outputs = layer(
                x,
                attention_mask=None,  # Assuming no mask for self-attention in decoder layers
                encoder_hidden_states=utterance_embeddings,

            )


            x = layer_outputs[0]

            discourse_attn_output, _ = self.discourse_cross_attention[i](query=x,key=discourse_embeddings,value=discourse_embeddings)

            # Combine utterance, discourse, and action information
            x = x + self.alpha * discourse_attn_output

        return x







class EncoderDecoder(nn.Module):
  def __init__(self, in_channels, out_channels, num_relations, num_classes):
    super(EncoderDecoder, self).__init__()
    self.inout = in_channels+out_channels
    self.speak_class = 4
    #Might as well use LSTM
    self.edge_attr_mlp = nn.Sequential(
            nn.Linear(num_relations, 64),  # Example dimension
            nn.Tanh(),
            # nn.Linear(32, 64),
            # nn.ReLU(),
            nn.Linear(64, out_channels)
        )
    self.structured_encoder = nn.GRU(self.inout, self.inout, batch_first=True, bidirectional=True)

    self.gatconv_layer = GATConv(self.inout*2, self.inout)#attempt otherwise self.inout*2, self.inout is not good enough
    self.Deconfig = BartConfig.from_pretrained('facebook/bart-base')

    self.Deconfig.d_model = 512
    self.Deconfig.encoder_layers = 12
    self.Deconfig.decoder_layers= 12
    self.Deconfig.decoder_attention_heads = 16
    self.Deconfig.encoder_attention_heads = 16
    self.decoder = MultiGranularityDecoder(self.Deconfig, num_layers = 4)
    self.bn1 = BatchNorm(self.inout)

    self.fc_layer1 = nn.Linear(self.inout, 64)
    self.fc_layer2 = nn.Linear(64, num_classes)
    self.dropout = nn.Dropout(0.1)
    self.tanh = nn.Tanh()
    self.lkrelu = nn.LeakyReLU()
    self.relu = nn.ReLU()

  def forward(self, samples):

    text = samples["features"]

    raw_edge_attr = (samples["edge_attr"][0]).float()
    edge_ind = samples["edge_index"][0]

    processed_edge_attr = self.edge_attr_mlp(raw_edge_attr)


    utterance_embeddings =  text.squeeze(0)

    # Concat Bert Encodings, Edge Attributes Acquired from Linear Layers(it is implicit since some utterances can have relations with multiple utterances, while some have none), Speaker One Hot Node
    utterance_embeddings = torch.cat([utterance_embeddings, processed_edge_attr], dim=1)

    structured_utterance_embeddings, _  = self.structured_encoder(utterance_embeddings)
    #utterance_embeddings = self.lkrelu(utterance_embeddings)

    discourse_embeddings = self.gatconv_layer(structured_utterance_embeddings, edge_ind)
    discourse_embeddings = self.bn1(discourse_embeddings)
    discourse_embeddings = self.relu(discourse_embeddings)

    summaries =  self.decoder(utterance_embeddings, discourse_embeddings, None)
    # Apply output layers to the summaries
    output = self.fc_layer1(summaries)
    output = self.tanh(output)
    output = self.dropout(output)
    output = self.fc_layer2(output)
    output = torch.sigmoid(output)
    output = output.squeeze(0).squeeze(1)

    return output



#based on observations that different percentile positions have greater prob of having important utts

def get_weight(index, total_length, filename):

    percentile = index / total_length
    if "a" in filename:
      if 0.02 <= percentile <= 0.18:
        return 0.11  # e.g., 2.0 or higher
      elif 0.24<=percentile<=0.36:
        return 0.0
      else:
        return 0.09
    if "b" in filename:
      if 0.0 <= percentile <= 0.38:
        return 0.09
      elif 0.38<=percentile<=0.94:
        return 0.08
      else:
        return 0.01
    if "c" in filename:
      if 0.0 <= percentile <= 0.18:
        return 0.1
      elif 0.18<=percentile<=0.66:
        return 0.07
      else:
        return 0.04

    if 0.0 <= percentile <= 0.2:
        return 0.1
    elif percentile<=0.82:
        return 0.07
    else:
        return 0.04

def weighted_loss(predictions, targets, dialogue_length, filename):
    loss = 0.0
    class_weights = torch.tensor([1.0,5.0])
    w_criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

    for i, (pred, target) in enumerate(zip(predictions, targets)):
        weight = 1.0 +get_weight(i, dialogue_length, filename)
        loss += weight * w_criterion(pred, target)
    return loss / len(predictions)

def train_model_here(model, optimizer, loss_criterion, num_epochs):
    iter = 0

    history_train_acc, history_val_acc, history_train_loss, history_val_loss = [], [], [], []
    best_f1 = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        for samples in train_loader:
            # Training mode

            model.train()


            labels = torch.tensor(samples["labels"]).float().to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            outputs = model(samples)


            # Calculate Loss: softmax --> cross entropy loss
            dialogue_length = len(samples["labels"])
            loss = loss_criterion(outputs, labels, dialogue_length, samples["filename"])


            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 50 == 0:
                # Get training statistics
                train_loss = loss.data.item()

                # Testing mode
                model.eval()
                # Calculate Accuracy
                correct = 0
                total = 0
                all_labels = []
                all_predictions = []
                # Iterate through test dataset
                iter_v =0
                for samples in val_loader:
                    # Load samples

                    labels = torch.tensor(samples["labels"]).float().to(device)

                    outputs = model(samples)


                    # Val loss

                    dialogue_length = len(samples["labels"])
                    val_loss = loss_criterion(outputs, labels, dialogue_length, samples["filename"])
                    #val_loss = criterion(outputs, labels)

                    predicted = outputs.ge(0.5).view(-1)
                    all_labels.extend(labels.type(torch.FloatTensor).cpu().numpy())
                    all_predictions.extend(predicted.type(torch.FloatTensor).cpu().numpy())

                    # Total number of labels
                    total += labels.size(0)

                    # Total correct predictions
                    correct += (predicted.type(torch.FloatTensor).cpu() == labels.type(torch.FloatTensor)).sum().item()



                accuracy = 100. * correct / total
                recall = recall_score(all_labels, all_predictions)*100.
                precision = precision_score(all_labels, all_predictions)*100.

                f1 = 2*recall*precision/(recall + precision)#f1_score(all_labels, all_predictions)*100.
                # Print Loss
                print('Iter: {} | Train Loss: {} | Val Loss: {} | Val Accuracy: {} | Recall: {} | Precision: {} | F1 Score : {}'.format(iter, train_loss, val_loss.item(), round(accuracy, 4), round(recall, 4), round(precision, 4), round(f1, 4)))

                # Append to history
                history_val_loss.append(val_loss.data.item())
                history_val_acc.append(round(accuracy, 2))
                history_train_loss.append(train_loss)

                # Save model when F1 Score beats best F1 Score
                if f1 > best_f1:
                    best_f1 = f1
                    # We can load this best model on the validation set later
                    torch.save(model.state_dict(), 'best_model.pth')
    return (history_train_acc, history_val_acc, history_train_loss, history_val_loss)





model = EncoderDecoder(in_channels=384, out_channels=128, num_relations=17, num_classes = 1)
class_weights = torch.tensor([1.0,6.0])
criterion = weighted_loss# Adjusted
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

(train_acc, val_acc, train_loss, val_loss) = train_model_here(model, optimizer, criterion, num_epochs=36)
plot_losses(train_loss, val_loss)




