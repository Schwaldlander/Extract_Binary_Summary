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

class UtteranceGraphDatasetTest(Dataset):
    def __init__(self, test_set,   relations_dict = relations_dict):
        super(UtteranceGraphDatasetTest, self).__init__(test_set)
        self.files = test_set
     
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

        transcription = test_coref[idx]


        npy_route = os.path.join(self.protxt, filename+'_ber.npy')
        if os.path.exists(npy_route):
            file_features = np.load(npy_route, allow_pickle=True)#
        else:
            for utterance in transcription:

              prepro_utter = data_preprocessor.preprocess(resolve_pronouns(utterance))
              utterance_list.append(prepro_utter)
            file_features = np.array(bert.encode(utterance_list))
            np.save(npy_route, file_features, allow_pickle=True)

        file_features = torch.tensor(file_features)

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

                if int(target)==len(transcription)-1:
                    # print('ZEND',int(target))
                    relationship_types.add('ZEND')
                    edges.append((int(target), int(target)))
                    edge_features.append(relation)

            # Create a one-hot encoding for relationship types



        encoded_edge_features = [self.relations_dict[rel] for rel in edge_features]

        # Convert to tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = F.one_hot(torch.tensor(encoded_edge_features, dtype=torch.long), num_classes=17)
        speak_index = []


        features = file_features



        return {"features":features,
            "edge_index":edge_index,
            "edge_attr":edge_attr,

            "filename":filename,
            "speaker_index":speak_index}





test_data = UtteranceGraphDatasetTest(test_set)
test_loader = DataLoader(test_data, batch_size=1, shuffle = True)
test_labels = {}
model = torch.load('best_model.pth')
for samples in test_loader:

  outputs = model(samples)

  outputs = outputs
  filename = samples["filename"][0]
  outputs = outputs.type(torch.FloatTensor).cpu().detach().int().numpy()
  test_labels[filename] = outputs.tolist()



with open("test_labels_text_gru_pro.json", "w") as file:
    json.dump(test_labels, file, indent=4)
