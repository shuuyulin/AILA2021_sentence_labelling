import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import math

class CLSlikeDataset(Dataset):
    def __init__(self, df, tokenized, seq_len=1):
        self.df = df
        self.tokenized = tokenized
        self.seq_len = seq_len
        self.max_sequ_len = 500
        self.max_curt_len = 500 if seq_len == 1 else int(self.max_sequ_len * 3 / 5) - 2
        self.max_side_len = 0 if seq_len == 1 else int(int(self.max_sequ_len * 2 / 5) / (seq_len - 1)) - 1

    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, idx):
        
        # Combine sentences
        # 101: CLS, 102: SEP, 0:PAD
        # new_id         = [CLS] + sentences[0] + [SEP] + sentences[1] + [SEP] + ... + sentences[n] + [SEP]
        # token_type_id  = 0, ..., 0, 1 ..., 1, 0 ..., 0
        # attention_mask = 1, ..., 1, 1 ..., 1, 1 ..., 1
        new_id = [101]
        token_type_id = [0]
        cur_docid = self.df['docid'][idx]
        for i in range(self.seq_len):
            j = idx - int(self.seq_len/2) + i
            if j < 0 or j >= len(self) or self.df['docid'][j] != cur_docid:
                new_id += [102]
                token_type_id += [0]
                continue

            sentence = self.tokenized[j]
                
            if i != int(self.seq_len/2):
                new_id += sentence.ids[:self.max_side_len]
                token_type_id += [0] * (len(sentence.ids[:self.max_side_len]) + 1)
            else:
                new_id += sentence.ids[:self.max_curt_len]
                token_type_id += [1] * (len(sentence.ids[:self.max_curt_len]) + 1)
            new_id += [102]

        attention_mask = [1] * len(new_id)

        # padding
        new_id         += [0] * (self.max_sequ_len - len(new_id))
        token_type_id  += [0] * (self.max_sequ_len - len(token_type_id))
        attention_mask += [0] * (self.max_sequ_len - len(attention_mask))
        
        return {
            'input_ids':torch.tensor(new_id),
            'token_type_ids':torch.tensor(token_type_id),
            'attention_mask':torch.tensor(attention_mask),
        }, torch.tensor(self.df['category'].iloc[idx])

class NERlikeDataset(Dataset):
    def __init__(self, df, tokenized, seq_len=1, isTrain=True, padding_threshold=0):
        self.df = df
        self.tokenized = tokenized
        self.seq_len = seq_len
        self.isTrain = isTrain
        self.max_sent_len = 501
        self.sent_len = self.max_sequ_len if seq_len == 1 else int(self.max_sent_len / seq_len) - 1
        self.padding_threshold = padding_threshold
        docid = set(df['docid'].tolist())
        self.doc_len = [len(df[df['docid'] == id].index) for id in docid]
            
        # idx of dataset to idx of tokenized is an abnormal mapping relation
        self.idx2tokidx = {}

        l_idx = 0 # last index of tokenized
        for idx in range(len(self)):
            d_idx = 0 # head index of document of dataset
            f_idx = 0 # first index of tokenized
            d_l_idx = 0  # last index of document of tokenized
            for i in self.doc_len:
                d_l_idx += i
                tmp = max(1, i - self.padding_threshold) if isTrain else int(math.ceil(i / self.seq_len))
                if d_idx + tmp > idx:
                    if isTrain:
                        f_idx += idx - d_idx 
                    else:
                        f_idx = l_idx
                    l_idx = min(f_idx + seq_len, d_l_idx)
                    break
                d_idx += tmp
                f_idx += i
                
            self.idx2tokidx[idx] = (f_idx, l_idx)
    
    def __len__(self):
        if self.isTrain:
            return sum([max(1, i - self.padding_threshold) for i in self.doc_len])
        else:
            return sum([int(math.ceil(i / self.seq_len)) for i in self.doc_len])
    
    def __getitem__(self, idx):
        # Combine sentences
        # 101: CLS, 102: SEP, 0:PAD
        # new_id         = [CLS] + sentences[0] + [SEP] + sentences[1] + [SEP] + ... + sentences[9] + [SEP]
        # token_type_id  = 0, 0, 0... 0
        # attention_mask = 1, 1, 1... 1
        
        f_idx, l_idx = self.idx2tokidx[idx]
        
        new_id, sent_pos = [101], [0]
        for idx, sentence in enumerate(self.tokenized[f_idx: l_idx]):
            new_id += sentence.ids[:self.sent_len] + [102]
            sent_pos += [sent_pos[-1] + len(sentence.ids[:self.sent_len]) + 1]
        
        target         = self.df['category'][f_idx : l_idx].tolist() # ckeck #TODO
        sent_mask      = [1] * (l_idx - f_idx)
        
        # sent padding
        new_id += [102] * (self.seq_len - (l_idx - f_idx))
        for _ in range(self.seq_len - len(sent_pos)): sent_pos += [sent_pos[-1] + 1]
        target += [-100] * (self.seq_len - len(target)) # arbitrary value to be ignore by crossentropy
        sent_mask += [0] * (self.seq_len - len(sent_mask))
        
        
        token_type_id  = [0] * len(new_id)
        attention_mask = [1] * len(new_id)
        
        # token padding
        new_id         += [0] * (self.max_sent_len - len(new_id))
        token_type_id  += [0] * (self.max_sent_len - len(token_type_id))
        attention_mask += [0] * (self.max_sent_len - len(attention_mask))
        
        sent_pos = sent_pos[:self.seq_len]
        
        assert np.shape(new_id) == (self.max_sent_len,), f'input_ids shape not match! {np.shape(new_id)}'
        assert np.shape(token_type_id) == (self.max_sent_len,), f'input_ids shape not match! {np.shape(token_type_id)}'
        assert np.shape(attention_mask) == (self.max_sent_len,), f'input_ids shape not match! {np.shape(attention_mask)}'
        assert np.shape(sent_pos) == (self.seq_len,), f'input_ids shape not match! {np.shape(sent_pos)}'
        assert np.shape(target) == (self.seq_len,), f'input_ids shape not match! {np.shape(target)}'
        assert np.shape(sent_mask) == (self.seq_len,), f'input_ids shape not match! {np.shape(sent_mask)}'
        
        return {
            'input_ids':torch.tensor(new_id),
            'token_type_ids':torch.tensor(token_type_id),
            'attention_mask':torch.tensor(attention_mask),
            'sent_pos':torch.tensor(sent_pos),
            'target':torch.tensor(target) # seq_len
        }, torch.tensor(sent_mask)