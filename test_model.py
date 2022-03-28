# not yet complete

from transformers import BertForSequenceClassification, BertTokenizerFast
from .BERT_base.model import NERlikeBERTClassifier
from .BERT_base.dataset import CLSlikeDataset, NERlikeDataset
from .BERT_BiLSTM_CRF.model import BiLSTM_CRF
from .BERT_BiLSTM_CRF.dataset import HierBERTDataset
from .utils import *
import torch
import os
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Configuration
cfg = {}
cfg['BERT_model'] = 'nlpaueb/legal-bert-base-uncased'
cfg['model_name'] = 'BERT_BiLSTM_CRF' # 'c' or 'BERT_CLSlike' or 'BERT_NERlike'
cfg['embed_from'] = 'mean' # mean or cls or mean_fined_BERT
cfg['batch_size'] = 2
cfg['seq_len'] = 658 # number of input sentences
cfg['padding_threshold'] = 0 # not used
cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"

# Path
BASEPATH = os.path.dirname(__file__)
TESTPATH = os.path.join(BASEPATH, './processed_data/test_data.csv')
TESTEMBPATH = os.path.join(BASEPATH, f'./processed_data/test_bert_embedding_{cfg["embed_from"]}.pkl')
# TESTPATH = os.path.join(BASEPATH, './processed_data/train_data.csv')
# TESTEMBPATH = os.path.join(BASEPATH, f'./processed_data/train_bert_embedding_{cfg["embed_from"]}.pkl')
PADEMBPATH = os.path.join(BASEPATH, f'./processed_data/PAD_embedding_{cfg["embed_from"]}.pkl')
CATNAMEPATH = os.path.join(BASEPATH, './processed_data/catagories_name.json')
MODELPATH = os.path.join(BASEPATH, './12best_model.pth')
# Read files
test_df = pd.read_csv(TESTPATH)
classes, num_class = read_classes(CATNAMEPATH)

with open(TESTEMBPATH, 'rb') as f:
    test_bert_embedding = pickle.load(f)
with open(PADEMBPATH, 'rb') as f:
    empty_bert_embedding = pickle.load(f)

# Tokenize sentence
tokenizer = BertTokenizerFast.from_pretrained(cfg['BERT_model'])
test_tokenized = tokenizer(test_df['sentence'].tolist(), add_special_tokens=False)

if cfg['model_name'] == 'BERT_CLSlike':
    model = BertForSequenceClassification.from_pretrained(cfg['BERT_model'],
                            num_labels=num_class,
                            output_attentions = False,
                            output_hidden_states = False,
                                                    ).to(device=cfg['device'])
    test_set = CLSlikeDataset(test_df, test_tokenized, seq_len=cfg['seq_len'])
elif cfg['model_name'] == 'BERT_NERlike':
    model = NERlikeBERTClassifier(cfg['BERT_model'], num_class, freeze_bert=False).to(device=cfg['device'])
    # test_set = NERlikeDataset
elif cfg['model_name'] == 'BERT_BiLSTM_CRF':
    test_doc_len = test_df.groupby('docid').size().values.tolist()
    model = BiLSTM_CRF(num_class=num_class, hidden_dim=128).to(cfg['device'])
    test_set = HierBERTDataset(test_df, test_bert_embedding, empty_bert_embedding, \
            test_doc_len, cfg['padding_threshold'], cfg['seq_len'], isTrain=False)

test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False)

model.load_state_dict(torch.load(MODELPATH))

model.eval()
totalloss, bestloss, totalacc, bestacc = 0, 10, 0, 0
predlist = []
with torch.no_grad():
    with tqdm(test_loader, unit='batch',desc='Test') as tqdm_loader:
        if cfg['model_name'] == 'BERT_CLSlike':
            for data, _ in tqdm_loader:
                for k in data:
                    data[k] = data[k].to(cfg['device'])
                output = model(**data)
                output = output.logits.cpu()
                pred = torch.argmax(output, dim=1)
                predlist.append(pred.tolist())
                
        elif cfg['model_name'] == 'BERT_NERlike':
            pass
        elif cfg['model_name'] == 'BERT_BiLSTM_CRF':
            for data in tqdm_loader:
                for k in data:
                    data[k] = data[k].to(cfg['device'])
                    
                output, _ = model(**data)
                output = output.cpu()[data['sent_mask'].cpu() != False]
                predlist.append(output.tolist())
            
# print(predlist)
pred = np.concatenate(np.array(predlist, dtype=object), axis=0)

# print(pred)
from sklearn.metrics import accuracy_score, confusion_matrix
# calculate accuracy
docid = set(test_df['docid'].tolist())
ttdacc = 0
allacc = 0
alltt = 0
for id in docid:
    idxrange = test_df.index[test_df['docid'] == id].tolist()
    acc = accuracy_score(test_df['category'][idxrange].tolist(), pred[idxrange].tolist())
    print(f'Document {id:0>2d} acc: {acc:.4f}')
    ttdacc += acc
    allacc += acc * len(pred[idxrange])
    alltt += len(pred[idxrange])
print(f'Average acc over documents: {ttdacc/len(docid):.4f}')
print(f'Average acc of all sentences:  {allacc/alltt:.4f}')
# confusion matrix

con_matrix = confusion_matrix(test_df['category'].tolist(), pred.tolist())
plot_confusion_matrix(con_matrix, os.path.join(BASEPATH, 'confusion_matrix.png'))