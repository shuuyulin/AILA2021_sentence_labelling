
from ..utils import *
from .dataset import NERlikeDataset
from .model import NERlikeBERTClassifier
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset 
from transformers import BertTokenizerFast
from tqdm.auto import tqdm

# Configurations
cfg = {}
cfg['model_name'] = 'nlpaueb/legal-bert-base-uncased'
cfg['record'] = 19
cfg['batch_size'] = 4
cfg['epoch'] = 7
cfg['lr'] = 1e-5
cfg['seq_len'] = 5 # number of input sentences
cfg['device'] =  "cuda" if torch.cuda.is_available() else "cpu"

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
    train_set = NERlikeDataset(train_df, train_tokenized, seq_len=cfg['seq_len'])
    valid_set = NERlikeDataset(valid_df, valid_tokenized, seq_len=cfg['seq_len'], isTrain=False)
    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=cfg['batch_size'], shuffle=False)
    # print(len(train_set))
    # print(len(valid_set))
    # print(valid_set[517])
    
    # Model / optimizer / scheduler
    model = NERlikeBERTClassifier(cfg['model_name'], num_class, freeze_bert=False).to(device=cfg['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], eps=1e-8)
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.1, min_lr=cfg['lr']*0.01)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, steps_per_epoch=len(train_loader), epochs=cfg['epoch'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= 2, eta_min=1e-6)

    tr_losses, vl_losses, tr_acces, vl_acces, lrs = [], [], [], [], []
    bestloss, bestacc, bepoch = 100, 0, 0
    # Running epoch
    for epoch in range(cfg['epoch']):
        print(f'epoch: {epoch}')
        tr_loss, tr_acc = train_one(model, train_loader, optimizer, criterion, scheduler)
        vl_loss, vl_acc = valid_one(model, valid_loader, criterion, valid_df)
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
        for idx, (data, sent_mask) in enumerate(tqdm_loader):
            for d in data:
                data[d] = data[d].to(cfg['device'])

            output, loss = model(**data)

            optimizer.zero_grad()
            loss.backward()
                        
            optimizer.step()
            # scheduler.step()

            nowloss = loss.item()

            pred = torch.argmax(output.cpu(), dim=2)
            acc = acc_counting(pred, data['target'].cpu(), sent_mask)

            totalloss += nowloss
            totalacc += acc
            tqdm_loader.set_postfix(loss=nowloss, avgloss=totalloss/(idx+1), avgACC=totalacc/(idx+1))
    return totalloss/len(tqdm_loader), totalacc/len(tqdm_loader)

def valid_one(model, dataloader, criterion, valid_df):
    model.eval()
    totalloss, bestloss, totalacc, bestacc = 0, 10, 0, 0
    predlist, masklist = [], []
    with torch.no_grad():
        with tqdm(dataloader,unit='batch',desc='Valid') as tqdm_loader:
            for idx, (data, sent_mask) in enumerate(tqdm_loader):
                for d in data:
                    data[d] = data[d].to(cfg['device'])

                output, loss = model(**data)

                nowloss = loss.item()

                pred = torch.argmax(output.cpu(), dim=2)
                predlist.append(pred)
                masklist.append(sent_mask)
                
                acc = acc_counting(pred, data['target'].cpu(), sent_mask)

                totalloss += nowloss
                totalacc += acc

                tqdm_loader.set_postfix(loss=nowloss, avgloss=totalloss/(idx+1) ,avgACC=totalacc/(idx+1))
                
    pred = np.concatenate(np.array(predlist, dtype=object), axis=0)
    mask = np.concatenate(np.array(masklist, dtype=object), axis=0)
    
    mic_acc = acc_counting(np.reshape(pred, (-1)), valid_df['category'].tolist(), np.reshape(mask, (-1)))
    # print(mic_acc)
    # avgloss = totalloss/len(tqdm_loader)
    return totalloss/len(tqdm_loader), mic_acc
if __name__ == '__main__':
    main()