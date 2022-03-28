import torch
import torch.nn as nn
from transformers import BertModel

class NERlikeBERTClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, model_name, num_class, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(NERlikeBERTClassifier, self).__init__()

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(model_name)

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, num_class),
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, token_type_ids, attention_mask, sent_pos, target=None):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor
                        with shape (batch_size, seq_len, max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask information
                        with shape (batch_size, seq_len, max_length)
        @return   output (torch.Tensor): an output tensor
                        with shape (batch_size, seq_len, num_labels)
        @return   loss (torch.Tensor): a tensor
                        with shape (batch_size, seq_len)
        """
        # Feed input to BERT
        bert_embedding = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,)

        # print(sent_pos)
        # Extract the last hidden state of the tokens in sent_pos
        dummy = sent_pos.unsqueeze(2).expand(sent_pos.size(0),sent_pos.size(1),bert_embedding['last_hidden_state'].size(2))
        sent_pos_hidden = torch.gather(bert_embedding['last_hidden_state'], dim=1, index=dummy)
        # batch, seq_len, 768

        # Feed input to classifier to compute logits
        output = self.classifier(sent_pos_hidden) #batch, seq_len, num_class
        if target != None:
            pred = output.permute(0, 2, 1) #batch, num_class, seq_len
            # target = target.permute(0, 2, 1) #batch, seq_len
            loss = self.criterion(pred, target)
            return output, loss
        else:
            return output 