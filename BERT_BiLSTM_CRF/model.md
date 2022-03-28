# BERT-BiLSTM-CRF

## pipeline
1. fine-tuning BERT with prediction task  
from ```BERT_CLSlike.py``` with seq_len = 1
2. sentence embedding with fine-tuned BERT  
file ```BERT_embedding.py```
3. train and test BiLSTM-CRF model with embedding sentences  
file ```main.py```