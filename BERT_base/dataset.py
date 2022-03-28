import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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
    def __init__(self, df, tokenized, seq_len=1):
        self.df = df
        self.tokenized = tokenized
        self.seq_len = seq_len
        self.max_sent_len = 501
        self.sent_len = self.max_sequ_len if seq_len == 1 else int(self.max_sent_len / seq_len) - 1

    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, idx):
        # Combine sentences
        # 101: CLS, 102: SEP, 0:PAD
        # new_id         = [CLS] + sentences[0] + [SEP] + sentences[1] + [SEP] + ... + sentences[9] + [SEP]
        # token_type_id  = 0, 0, 0... 0
        # attention_mask = 1, 1, 1... 1
        new_id, token_type_id, sent_pos, target, sent_mask = [101], [], [0], [], []
        cur_docid = self.df['docid'][idx]
        for i in range(self.seq_len):
            j = idx - int(self.seq_len/2) + i
            if j < 0 or j >= len(self) or self.df['docid'][j] != cur_docid:
                new_id += [102]
                token_type_id += [0]
                sent_pos += [sent_pos[-1] + 1] if i != self.seq_len - 1 else []
                
                target += [0] # arbitrary value, will be masked when calculating accuracy
                sent_mask += [0]
                continue
            
            sentence = self.tokenized[j]
            new_id += sentence.ids[:self.sent_len] + [102]
            target += [self.df['category'][j].tolist()]
            sent_mask += [1]
            if i != self.seq_len - 1:
                sent_pos += [sent_pos[-1] + len(sentence.ids[:self.sent_len]) + 1]
            
        token_type_id  = [0] * len(new_id)
        attention_mask = [1] * len(new_id)
        
        # padding
        new_id         += [0] * (self.max_sent_len - len(new_id))
        token_type_id  += [0] * (self.max_sent_len - len(token_type_id))
        attention_mask += [0] * (self.max_sent_len - len(attention_mask))
        
        return {
            'input_ids':torch.tensor(new_id),
            'token_type_ids':torch.tensor(token_type_id),
            'attention_mask':torch.tensor(attention_mask),
            'sent_pos':torch.tensor(sent_pos),
            'target':torch.tensor(target) # seq_len
        }, torch.tensor(sent_mask)