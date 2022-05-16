# AILA 2020 task 1 rhetorical labelling  

[competition source](https://sites.google.com/view/aila-2021)  
法律援助人工智能 (AILA) 是一系列法律相關共享任務，旨在開發數據集和方法來解決各種法律信息學問題。  

## Task  
此任務旨在對法律文件進行語意分割，具體而言這是一個句子分類任務，每一個句子必須被分配給七個預定義的修辭角色，包涵 Facts、Ruling by Lower Court、Argument、Statute、Precedent、Ratio of the decision、Ruling by the Present Court。

<img src="https://user-images.githubusercontent.com/56257705/168584241-1a11ff1d-c386-4e66-9bb8-3f0412947007.png" width="500" />  

## Dataset
- 60 traning documents  
  -  split to 48:12 training:validation  
-  10 testing documents  

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

<table><tr>
    <td><img src="https://i.imgur.com/sXJci2g.png" width="200" /></td>
    <td><img src="https://i.imgur.com/yLRDLqS.png" width="200" /></td>
    <tr>
        <td><center>BERT single sequence labelling</center></td>  
        <td><center>BERT multiple sequence labelling</center></td>  
    </tr>  
</tr></table>  

- BERT-BiLSTM-CRF  
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
| BERT-BiLSTM-CRF fine-tuned | 0.739 | 0.695 |  
