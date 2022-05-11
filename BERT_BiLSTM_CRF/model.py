import torch
import torch.nn as nn
from torchcrf import CRF
# model with pytorch-crf module

class BiLSTM_CRF(nn.Module):
    """Bert Model for sentence classification task
    """
    def __init__(self, num_class, lstm_layer=1, hidden_dim=128, dropout_rate=0.5):
        """
        @param    classifier: a torch.nn.Module classifier
        """
        super(BiLSTM_CRF, self).__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
    
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=lstm_layer,\
            batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(hidden_dim * 2)
        
        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Linear(hidden_dim * 2, num_class)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.crf = CRF(num_class, batch_first=True)
        self.crf.reset_parameters()
        
    def forward(self, bert_embedding, sent_mask, target=None):
        """
        Feed BERT_embedding to LSTM, CRF and the classifier to compute output probability distributions.
        @param    bert_embedding (torch.Tensor): an input tensor
                        with shape (batch_size, seq_len, 768)
        @param    mask (torch.Tensor.bool): an input tensor
                        with shape (batch_size, seq_len)
        @param    target (torch.Tensor): an input tensor
                        with shape (batch_size, seq_len)
        @return   output (torch.Tensor): an output tensor
                        with shape (batch_size, seq_len(masked))
        @return   loss (torch.Tensor): L = crf loss
                        with shape ()
        """
        _, seq_len, _ = bert_embedding.shape
        # Feed input to LSTM
        lstm_output, _ = self.lstm(bert_embedding) # (batch, seq_len, hidden_dim * 2)
        
        # Feed lstm output to ln and classifier
        
        # t_mask = sent_mask.unsqueeze(-1).expand(sent_mask.shape[0], sent_mask.shape[1], lstm_output.shape[2])
        lstm_output = self.ln(lstm_output) # (batch, seq_len, hidden_dim * 2)
        lstm_output = self.dropout(lstm_output)
        emit_score = self.classifier(lstm_output) # (batch, seq_len, num_class)

        crf_output = self.crf.decode(emit_score, sent_mask) # list, shape(B, seq_len(masked))
        
        # full 0 in masked crf output
        crf_output = torch.Tensor([i + [0] * (seq_len - len(i)) for i in crf_output])
#         display(crf_output.shape)
        if target != None:
            loss = -self.crf(emit_score, target, sent_mask, reduction='mean')
            return crf_output, loss
        else:
            return crf_output