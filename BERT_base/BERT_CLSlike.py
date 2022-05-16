
from ..utils import *
from .dataset import CLSlikeDataset
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset 
from transformers import BertForSequenceClassification, BertTokenizerFast
from tqdm.auto import tqdm
from ranger21 import Ranger21

# Config
cfg = {}
cfg['model_name'] = 'nlpaueb/legal-bert-base-uncased'
cfg['batch_size'] = 4
cfg['record'] = 19
cfg['epoch'] = 4
cfg['lr'] = 1e-5
cfg['seq_len'] = 5
cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
BASEPATH = os.path.dirname(__file__)
TRAINPATH = os.path.join(BASEPATH, '../processed_data/train_data.csv')
VALIDPATH = os.path.join(BASEPATH, '../processed_data/valid_data.csv')
CATNAMEPATH = os.path.join(BASEPATH, '../processed_data/catagories_name.json')
MODELPATH = os.path.join(BASEPATH, './best_model.pth')
RECORDPATH = os.path.join(BASEPATH, f'../record/{cfg["record"]}/')

if not os.path.isdir(RECORDPATH):
    os.mkdir(RECORDPATH)
MODELPATH = os.path.join(RECORDPATH, f'{cfg["record"]}best_model.pth')

# Fix random seed for reproducibility
same_seeds(0)

def main():
    # Read files
    train_df = pd.read_csv(TRAINPATH)
    valid_df = pd.read_csv(VALIDPATH)
    classes, num_class = read_classes(CATNAMEPATH)

    # Tokenize sentence
    tokenizer = BertTokenizerFast.from_pretrained(cfg['model_name'])

    train_tokenized = tokenizer(train_df['sentence'].tolist(), add_special_tokens=False)
    valid_tokenized = tokenizer(valid_df['sentence'].tolist(), add_special_tokens=False)

    # Dataset / Dataloader
    train_set = CLSlikeDataset(train_df, train_tokenized, seq_len=cfg['seq_len'])
    valid_set = CLSlikeDataset(valid_df, valid_tokenized, seq_len=cfg['seq_len'])
    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=cfg['batch_size'], shuffle=False)

    # Model / optimizer / scheduler
    model = BertForSequenceClassification.from_pretrained(cfg['model_name'],
                            num_labels=num_class,
                            output_attentions = False,
                            output_hidden_states = False,
                            # attention_probs_dropout_prob=0.5,
                            # hidden_dropout_prob=0.5
                                                    ).to(device=cfg['device'])

    # optimizer = Ranger21(model.parameters(), lr=cfg['lr'], num_epochs=cfg['epoch'], num_batches_per_epoch=cfg['batch_size'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], eps=1e-8)
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.1, min_lr=cfg['lr']*0.01)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, steps_per_epoch=len(train_loader), epochs=cfg['epoch'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= 2, eta_min=1e-6)
    criterion = torch.nn.CrossEntropyLoss()

    tr_losses, vl_losses, tr_acces, vl_acces, lrs = [], [], [], [], []
    bestloss, bestacc, bepoch = 100, 0, 0
    # Running epoch
    for epoch in range(cfg['epoch']):
        print(f'epoch: {epoch}')
        tr_loss, tr_acc = train_one(model, train_loader, optimizer, criterion, scheduler)
        vl_loss, vl_acc = valid_one(model, valid_loader, criterion)
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

    # Plot train/valid loss, accuracy, learning rate
    plot_fg(tr_losses, 'losses', 'loss', RECORDPATH, vl_losses)
    plot_fg(tr_acces, 'acces', 'acc', RECORDPATH, vl_acces)
    plot_fg(lrs, 'lrs', 'lr', RECORDPATH)

def train_one(model, dataloader, optimizer, criterion, scheduler):
    model.train()
    totalloss=0
    totalacc=0
    with tqdm(dataloader, unit='batch', desc='Train') as tqdm_loader:
        for idx, (data, label) in enumerate(tqdm_loader):
            for d in data:
                data[d] = data[d].to(cfg['device'])

            output = model(**data)
            
            output = output.logits.cpu()
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
                        
            optimizer.step()
            # scheduler.step()

            nowloss = loss.item()

            pred = torch.argmax(output, dim=1)
            acc = acc_counting(pred, label.cpu())

            totalloss += nowloss
            totalacc += acc
            tqdm_loader.set_postfix(loss=nowloss, avgloss=totalloss/(idx+1), avgACC=totalacc/(idx+1))
    return totalloss/len(tqdm_loader), totalacc/len(tqdm_loader)

def valid_one(model, dataloader, criterion):
    model.eval()
    totalloss, totalacc, bestloss, bestacc = 0, 0, 10, 0
    with torch.no_grad():
        with tqdm(dataloader,unit='batch',desc='Valid') as tqdm_loader:
            for idx, (data, label) in enumerate(tqdm_loader):
                for d in data:
                    data[d] = data[d].to(cfg['device'])

                output = model(**data)
                
                output = output.logits.cpu()
                loss = criterion(output, label)

                nowloss = loss.item()

                pred = torch.argmax(output, dim=1)
                acc = acc_counting(pred, label.cpu())

                totalloss += nowloss
                totalacc += acc
                tqdm_loader.set_postfix(loss=nowloss, avgloss=totalloss/(idx+1) ,avgACC=totalacc/(idx+1))
    return totalloss/len(tqdm_loader), totalacc/len(tqdm_loader)

if __name__ == '__main__':
    main()

