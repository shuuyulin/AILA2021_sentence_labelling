from ..utils import *
from .dataset import HierBERTDataset
from .model import BiLSTM_CRF
import numpy as np
import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader
from ranger21 import Ranger21
from ranger import Ranger
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

# Configuration
cfg = {}
cfg['embed_from'] = 'mean_fined_BERT' # mean or cls or mean_fined_BERT (fine-tuned by same prediction task)
cfg['record'] = 19
cfg['batch_size'] = 16
cfg['epoch'] = 20
cfg['lr'] = 1e-5
cfg['seq_len'] = 658 # number of input sentences
cfg['padding_threshold'] = 0 # max number of sentence padding
cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"
cfg['dropout_rate'] = 0.5
cfg['lstm_layer'] = 1

# Paths
BASEPATH = os.path.dirname(__file__)
TRAINPATH = os.path.join(BASEPATH, '../processed_data/train_data.csv')
VALIDPATH = os.path.join(BASEPATH, '../processed_data/valid_data.csv')
TRAINEMBPATH = os.path.join(BASEPATH, f'../processed_data/train_bert_embedding_{cfg["embed_from"]}.pkl')
VALIDEMBPATH = os.path.join(BASEPATH, f'../processed_data/valid_bert_embedding_{cfg["embed_from"]}.pkl')
PADEMBPATH = os.path.join(BASEPATH, f'../processed_data/PAD_embedding_{cfg["embed_from"]}.pkl')
CATNAMEPATH = os.path.join(BASEPATH, '../processed_data/catagories_name.json')
RECORDPATH = os.path.join(BASEPATH, f'../record/{cfg["record"]}/')

if not os.path.isdir(os.path.join(BASEPATH, f'../record{cfg["record"]}')):
    os.mkdir(os.path.join(RECORDPATH, str(cfg["record"])))
MODELPATH = os.path.join(RECORDPATH, f'./{cfg["record"]}best_model.pth')

# Fix random seed for reproducibility
same_seeds(0)

def main():

    # Read files
    train_df = pd.read_csv(TRAINPATH)
    valid_df = pd.read_csv(VALIDPATH)
    classes, num_class = read_classes(CATNAMEPATH)

    with open(TRAINEMBPATH, 'rb') as f:
        train_bert_embedding = pickle.load(f)
    with open(VALIDEMBPATH, 'rb') as f:
        valid_bert_embedding = pickle.load(f)
    with open(PADEMBPATH, 'rb') as f:
        empty_bert_embedding = pickle.load(f)

    # Document number of sentence
    train_doc_len = train_df.groupby('docid').size().values.tolist()
    valid_doc_len = valid_df.groupby('docid').size().values.tolist()

    # Dataset / Dataloader
    train_set = HierBERTDataset(train_df, train_bert_embedding, empty_bert_embedding, \
            train_doc_len, cfg['padding_threshold'], cfg['seq_len'], isTrain=True)
    valid_set = HierBERTDataset(valid_df, valid_bert_embedding, empty_bert_embedding, \
            valid_doc_len, cfg['padding_threshold'], cfg['seq_len'], isTrain=False)
    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=cfg['batch_size'], shuffle=False)

    # model, optimizer, scheduler
    model = BiLSTM_CRF(num_class=num_class, lstm_layer=cfg['lstm_layer'],\
        hidden_dim=128, dropout_rate=cfg['dropout_rate']).to(cfg['device'])

    optimizer = Ranger21(model.parameters(), lr=cfg['lr'], num_epochs=cfg['epoch'], num_batches_per_epoch=len(train_loader))
    # optimizer = Ranger(model.parameters())
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], eps=1e-8)
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9)
    # scheduler = None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.1, min_lr=cfg['lr']*1e-10)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, steps_per_epoch=len(train_loader), epochs=cfg['epoch'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= 2, eta_min=1e-6)

    tr_losses, vl_losses, tr_acces, vl_acces, lrs = [], [], [], [], []
    bestloss, bestacc, bepoch = 100, 0, 0
    # Run epochs
    for epoch in range(cfg['epoch']):
        print(f'epoch: {epoch}')
        tr_loss, tr_acc = train_one(model, train_loader, optimizer, scheduler)
        vl_loss, vl_acc = valid_one(model, valid_loader)
        lrs.append(get_lr(optimizer))
        scheduler.step(vl_acc)
        tr_losses.append(tr_loss)
        vl_losses.append(vl_loss)
        tr_acces.append(tr_acc)
        vl_acces.append(vl_acc)
        if vl_acc >= bestacc:
            bestacc = vl_acc
            bepoch = epoch
        if vl_loss <= bestloss:
            bestloss = vl_loss
            torch.save(model.state_dict(), MODELPATH)
            
    print(f'best valid acc: {bestacc}, epoch: {bepoch}')
    
    # Plot figure of loss, accuracy and learning rate
    plot_fg(tr_losses, 'losses', 'loss', RECORDPATH, vl_losses)
    plot_fg(tr_acces, 'acces', 'acc', RECORDPATH, vl_acces)
    plot_fg(lrs, 'lrs', 'lr', RECORDPATH)

def train_one(model, dataloader, optimizer, scheduler):
    model.train()
    totalloss=0
    totalacc=0
    with tqdm(dataloader, unit='batch', desc='Train') as tqdm_loader:
        for idx, data in enumerate(tqdm_loader):
            for k in data:
                data[k] = data[k].to(cfg['device'])
            
            output, loss = model(**data)

            optimizer.zero_grad()
            loss.backward()
    
            optimizer.step()
            # scheduler.step()
            
            prednum = torch.sum(data['sent_mask'].cpu()).detach()
            nowloss = loss.item() / prednum # loss is mean over batchs as torch-crf attibute: reduciton='mean'

            acc = acc_counting(output.cpu(), data['target'].cpu(), data['sent_mask'].cpu())

            totalloss += nowloss
            totalacc += acc
            tqdm_loader.set_postfix(loss=nowloss, avgloss=totalloss/(idx+1), avgACC=totalacc/(idx+1))
    return totalloss/len(tqdm_loader), totalacc/len(tqdm_loader)

def valid_one(model, dataloader):
    model.eval()
    totalloss, totalacc, totalprednum = 0, 0, 0
    
    with torch.no_grad():
        with tqdm(dataloader,unit='batch',desc='Valid') as tqdm_loader:
            for idx, data in enumerate(tqdm_loader):
                for k in data:
                    data[k] = data[k].to(cfg['device'])
            
                output, loss = model(**data)
                
                prednum = torch.sum(data['sent_mask'].cpu()).detach()
                
                nowloss = loss.item() / prednum
                acc = acc_counting(output.cpu(), data['target'].cpu(), data['sent_mask'].cpu())
                
                totalprednum += prednum
                totalloss += nowloss * prednum
                totalacc += acc * prednum
                
                tqdm_loader.set_postfix(loss=nowloss, avgloss=totalloss/(totalprednum) ,avgACC=totalacc/(totalprednum))
                
    return totalloss/totalprednum, totalacc/totalprednum
if __name__ == '__main__':
    main()