import numpy as np
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset 

class HierBERTDataset(Dataset):
    def __init__(self, df, embedding, pad_bert_embedding, doc_len, padding_threshold, seq_len=1, isTrain=True):
        self.df = df
        self.embedding = embedding
        self.pad_bert_embedding = pad_bert_embedding.tolist()
        self.doc_len = doc_len
        self.seq_len = seq_len
        self.padding_threshold = padding_threshold
        self.isTrain = isTrain
        # idx to embedding idx is an abnormal mapping relation
        self.idx2embidx = {}

        for idx in range(len(self)):
            d_idx = 0 # head index of document of dataset
            f_idx = 0 # first index of embedding
            l_idx = 0 # last index of embedding
            if isTrain:
                for i in self.doc_len:
                    if d_idx + max(1, i - self.padding_threshold) > idx:
                        f_idx += idx - d_idx
                        l_idx = f_idx + min(self.seq_len, i)
                        break
                    d_idx += max(1, i - self.padding_threshold)
                    f_idx += i
            else:
                for i in self.doc_len:
                    if d_idx + int(math.ceil(i / self.seq_len)) > idx:
                        f_idx += idx - d_idx
                        l_idx = f_idx + min(self.seq_len, i)
                        break
                    d_idx += int(math.ceil(i / self.seq_len))
                    f_idx += i
            self.idx2embidx[idx] = (f_idx, l_idx)
            
    def __len__(self):
        if self.isTrain:
            return sum([max(1, i - self.padding_threshold) for i in self.doc_len])
        else:
            return sum([int(math.ceil(i / self.seq_len)) for i in self.doc_len])
    
    def __getitem__(self, idx):
        '''
        @return bert_embedding          shape (seq_len, 768)
        @return sent_mask               shape (seq_len)
        '''
        f_idx, l_idx = self.idx2embidx[idx]

        embedding_seq = self.embedding[f_idx:l_idx]
        target = self.df['category'].iloc[f_idx:l_idx].tolist()
            
        sent_mask = [1] * len(embedding_seq)

        # Padding length to seq_len
        for _ in range(self.seq_len - len(embedding_seq)):
            embedding_seq = np.append(embedding_seq, self.pad_bert_embedding, axis=0)
        sent_mask += [0] * (self.seq_len - len(sent_mask))
        target += [0] * (self.seq_len - len(target)) #被mask不被訓練
        
        return {
            'bert_embedding' : torch.Tensor(embedding_seq),
            'sent_mask' : torch.Tensor(sent_mask).bool(),
            'target' : torch.Tensor(target).to(dtype=torch.int64)
        }