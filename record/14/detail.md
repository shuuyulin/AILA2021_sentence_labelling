# experiment details
file: BERT-BiLSTM-CRF

## configuration
```
cfg['batch_size'] = 4  
cfg['epoch'] = 30  
cfg['lr'] = 1e-3  
cfg['seq_len'] = 658  
cfg['padding_threshold'] = 0  
cfg['dropout_rate'] = 0.5 (between crf, lstm)  
```
model: Legal-BERT fine-tuned with same prediction task  
optimizer: Ranger21  
scheduler: CosineAnnealingLR  

valid best: about 0.66(not recorded)

## testing result

Document 01 acc: 0.6724  
Document 02 acc: 0.7812  
Document 03 acc: 0.7347  
Document 04 acc: 0.5778  
Document 05 acc: 0.4250  
Document 06 acc: 0.5422  
Document 07 acc: 0.5294  
Document 08 acc: 0.4177  
Document 09 acc: 0.6020  
Document 10 acc: 0.6726  
Average acc: 0.5955  

