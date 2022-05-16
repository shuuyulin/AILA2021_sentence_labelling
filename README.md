# AILA 2021 task 1 rhetorical labelling  

[AILA 2021](https://sites.google.com/view/aila-2021)  

Artificial Intelligence Legal Assistance is a series of shared tasks. It aimed at building datasets and resolving a variety of  legal informatic problems.  

## AILA 2021 task 1  
This task is to sematic segmentation a legal document. Specifically, it a sentence labelling task. Each sentence of a India Legal jedgement should be assigned to a "Rhetorical Role", including Facts, Ruling by Lower Court, Argument, Statute, Precedent, Ratio of the decision and Ruling by the Present Court.

<img src="https://user-images.githubusercontent.com/56257705/168584241-1a11ff1d-c386-4e66-9bb8-3f0412947007.png" width="500" />  

## Dataset
- 60 training documents  
  -  split to 48/12 training/validation  
- 10 testing documents  

## Data analyzation
- Categories Distribution Pie Chart  
<img src="https://user-images.githubusercontent.com/56257705/164142010-700b63f4-d799-4e17-96ac-05e946d4fd5e.png" width="200" />

- Number of tokens of sentences  
<table><tr>
    <td><img src="https://i.imgur.com/9LKGPqc.png" width="200" /></td>
    <td><img src="https://i.imgur.com/QL8F7u5.png" width="200" /></td>
    <tr>
        <td><center>train (max: 383)</center></td>  
        <td><center>valid (max: 286)</center></td>  
    </tr>  
</tr></table>  

- Number of sentences of documents  
<table><tr>
    <td><img src="https://i.imgur.com/ByC8aCB.png" width="200" /></td>
    <td><img src="https://i.imgur.com/RjnN870.png" width="200" /></td>
    <tr>
        <td><center>train (max: 658)</center></td>  
        <td><center>valid (max: 471)</center></td>  
    </tr>  
</tr></table>

## Models  
- 2 sentence-level fine-tuned BERT methods  
  - single sequence labelling: Predict middle sentence rhetorical role by sentences beside it.
  - multiple sequence labelling: Predict multiple sentences rhetorical role by themself.
<table><tr>
    <td><img src="https://i.imgur.com/sXJci2g.png" width="200" /></td>
    <td><img src="https://i.imgur.com/yLRDLqS.png" width="200" /></td>
    <tr>
        <td><center>BERT single sequence labelling</center></td>  
        <td><center>BERT multiple sequence labelling</center></td>  
    </tr>  
</tr></table>  

- BERT-BiLSTM-CRF  
  - After each sentence is embedded in the BERT sentence, the result of the last hidden layer is averaged to obtain the sentence vector of each sentence, and then the BiLSTM and CRF are used to predict the sentence category for multiple sentence vectors.  
  - The BERT part uses pretrained LEGAL-BERT, or fine-tuned LEGAL-BERT which is fine-tuned by the single-sentence labelling task.  
<img src="https://user-images.githubusercontent.com/56257705/164142250-e5fa90f8-0fb2-47c1-a237-d57fd4b786f6.png" width="500" />  

- fine-tuning LEGAL-BERT pipeline  
<img src="https://i.imgur.com/hHWazX5.png" width="500" />  

## Result  

- BERT v.s. Legal BERT

|       |  BERT | LegalBERT |
| ----- |:-----:| --------- |
| valid | 0.654 |   0.683   |  

- models comparison  

|                            | valid | test  |
| -------------------------- |:-----:| ----- |
| BERT single-seq labeling   | 0.672 | 0.697 |
| BERT multi-seq labeling    | 0.659 | 0.658 |
| BERT-BiLSTM-CRF            | 0.619 | 0.623 |
| BERT-BiLSTM-CRF fine-tuned | 0.663 | 0.695 |  

## How to use?
- training  
BERT single-seq labelling at ```BERT_base/BERT_CLS_like.py```  
BERT multi-seq labelling at ```BERT_base/BERT_NER_like.py```  
BERT-BiLSTM-CRF at ```BERT_BiLSTM_CRF/main.py```  
at the root folder of repo run ```python -m AILA2021_sentence_labelling.{file path}```  
- testing  
at the root folder of repo run ```python -m AILA2021_sentence_labelling.test_model```  

- make sure config of running file is correct
