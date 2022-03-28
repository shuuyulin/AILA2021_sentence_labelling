from ..utils import *
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizerFast, BertForSequenceClassification
from tqdm.auto import tqdm

# Configuration
cfg = {}
cfg['embed_from'] = 'mean_fined_BERT' # mean or cls or mean_fined_BERT (fine-tuned by same prediction task)
cfg['model_name'] = 'nlpaueb/legal-bert-base-uncased'
cfg['batch_size'] = 8
cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
BASEPATH = os.path.dirname(__file__)
CATNAMEPATH = os.path.join(BASEPATH, '../processed_data/catagories_name.json')
TRAINPATH = os.path.join(BASEPATH, '../processed_data/train_data.csv')
TRAINEMBPATH = os.path.join(BASEPATH, f'../processed_data/train_bert_embedding_{cfg["embed_from"]}.pkl')
VALIDPATH = os.path.join(BASEPATH, '../processed_data/valid_data.csv')
VALIDEMBPATH = os.path.join(BASEPATH, f'../processed_data/valid_bert_embedding_{cfg["embed_from"]}.pkl')
TESTPATH = os.path.join(BASEPATH, '../processed_data/test_data.csv')
TESTEMBPATH = os.path.join(BASEPATH, f'../processed_data/test_bert_embedding_{cfg["embed_from"]}.pkl')
PADEMBPATH = os.path.join(BASEPATH, f'../processed_data/PAD_embedding_{cfg["embed_from"]}.pkl')

MODELPATH = os.path.join(BASEPATH, './finetuned_BERT.pth')

# Read files
train_df = pd.read_csv(TRAINPATH)
valid_df = pd.read_csv(VALIDPATH)
test_df = pd.read_csv(TESTPATH)
classes, num_class = read_classes(CATNAMEPATH)

# Tokenize sentence
tokenizer = BertTokenizerFast.from_pretrained(cfg['model_name'])

train_tokenized = tokenizer(train_df['sentence'].tolist(), add_special_tokens=True, padding=True, truncation=True)
valid_tokenized = tokenizer(valid_df['sentence'].tolist(), add_special_tokens=True, padding=True, truncation=True)
test_tokenized = tokenizer(test_df['sentence'].tolist(), add_special_tokens=True, padding=True, truncation=True)

class bert_emb_dataset(Dataset):
    def __init__(self, data_tokenized, df):
        self.data = [{key : torch.Tensor(data_tokenized[key][i]).to(dtype=torch.int64) \
                      for key in data_tokenized.keys()} for i in range(len(data_tokenized['input_ids']))]
        self.df = df
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
            
        return self.data[idx]

# Dataset / Dataloader
emb_train_dataset = bert_emb_dataset(train_tokenized, train_df)
emb_valid_dataset = bert_emb_dataset(valid_tokenized, valid_df)
emb_train_loader = DataLoader(emb_train_dataset, batch_size=cfg['batch_size'], shuffle=False)
emb_valid_loader = DataLoader(emb_valid_dataset, batch_size=cfg['batch_size'], shuffle=False)

emb_test_dataset = bert_emb_dataset(test_tokenized, test_df)
emb_test_loader = DataLoader(emb_test_dataset, batch_size=cfg['batch_size'], shuffle=False)

# Instantiate BERT model
# bert = BertModel.from_pretrained(cfg['model_name']).to(device=cfg['device'])
bert = BertForSequenceClassification.from_pretrained(cfg['model_name'],
                            num_labels=num_class,
                            output_attentions = False,
                            output_hidden_states = True,
                                                    ).to(device=cfg['device'])
bert.load_state_dict(torch.load(MODELPATH))
bert.eval()

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(last_hidden, attention_mask):
    token_embeddings = last_hidden  #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def cls_pooling(last_hidden):
    # return last_hidden[: , 0, :] # B, 388, 768
    return last_hidden[:, 0, :]

def bert_embedding(model, loader):
    embedding_list = []

    with torch.no_grad():
        with tqdm(loader, unit='batch') as tqdm_loader:
            for idx, data in enumerate(tqdm_loader):
                for k in data:
                    data[k] = data[k].to(cfg['device'])
                
                model_output = model(**data)
                # for k in model_output:
                #     display(k, model_output[k].shape)
                # assert model_output.hidden_states[-1].shape == (cfg['batch_size'])),\
                #     f'shape error! hidden: {model_output.hidden_states[-1].shape}'
                    
                if cfg['embed_from'] == 'mean':
                    pooling_output = mean_pooling(model_output.hidden_states[-1], data['attention_mask'])
                else:
                    pooling_output = cls_pooling(model_output.hidden_states[-1])
    
                embedding_list.append(pooling_output.cpu())
    embedding_list = np.concatenate(embedding_list, axis=0)
    return embedding_list


train_bert_embedding = bert_embedding(bert, emb_train_loader)
# valid_bert_embediding = bert_embedding(bert, emb_valid_loader)
# test_bert_embeddng = bert_embedding(bert, emb_test_loader)

import pickle
with open(TRAINEMBPATH, 'wb') as f:
    pickle.dump(train_bert_embedding, f, pickle.HIGHEST_PROTOCOL)
# with open(VALIDEMBPATH, 'wb') as f:
#     pickle.dump(valid_bert_embedding, f, pickle.HIGHEST_PROTOCOL)
# with open(TESTEMBPATH, 'wb') as f:
#     pickle.dump(test_bert_embedding, f, pickle.HIGHEST_PROTOCOL)

def empty_seq_embedding(bert_input_max_len, bert, tokenizer):
    bert = bert.to('cpu')
    empty_sent = ''
    empty_tokenized = tokenizer(empty_sent, add_special_tokens=True,
                padding="max_length",  max_length=bert_input_max_len, return_tensors='pt')
    model_output = bert(**empty_tokenized)
    if cfg['embed_from'] == 'mean':
        return mean_pooling(model_output.hidden_states[-1], empty_tokenized['attention_mask'])
    else:
        return cls_pooling(model_output.hidden_states[-1])

# empty_tokenized = empty_seq_embedding(len(train_tokenized['input_ids'][0]), bert, tokenizer)

# with open(PADEMBPATH, 'wb') as f:
#     pickle.dump(empty_tokenized, f, pickle.HIGHEST_PROTOCOL)
