# Task 3: food taste similarity
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import numpy as np 
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ImageDataset
from model import SiameseResNet

DATADIR = './data/'
LEARNING_RATE = 1e-5
EPOCHS = 100
BATCH_SIZE = 128
PATIENCE = 3

# --------------- data
# prepare file train to split into train and validation
if not os.path.exists(f'{DATADIR}train_triplets_backup.txt'):
    train_triplets = pd.read_csv(f'{DATADIR}train_triplets.txt', sep=' ', header=None, dtype=str)
    train_triplets.to_csv(f'{DATADIR}train_triplets_backup.txt', sep=' ', header=None, index=False)

    train_triplets = pd.read_csv(f'{DATADIR}train_triplets_backup.txt', sep=' ', header=None, dtype=str)
    train_triplets, val_triplets = train_test_split(train_triplets, test_size=0.2)
    train_triplets.to_csv(f'{DATADIR}train_triplets.txt', sep=' ', header=None, index=False)
    val_triplets.to_csv(f'{DATADIR}val_triplets.txt', sep=' ', header=None, index=False)

# read image datasets
images_train = ImageDataset(datadir=DATADIR, split='train')
images_val = ImageDataset(datadir=DATADIR, split='val')
images_test = ImageDataset(datadir=DATADIR, split='test')

trainloader = DataLoader(images_train, batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(images_val, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(images_test, batch_size=BATCH_SIZE)

# --------------- siamese network
model = SiameseResNet()
criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1-F.cosine_similarity(x, y), reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.DataParallel(model)
model.to(device)

# initialize logs
logs = {'train': np.zeros(EPOCHS), 'val': np.zeros(EPOCHS)}

for epoch in tqdm(range(EPOCHS)):
    # fit model
    model.train()

    for anc_in, pos_in, neg_in in tqdm(trainloader):
        anc_in, pos_in, neg_in = anc_in.to(device), pos_in.to(device), neg_in.to(device)

        optimizer.zero_grad() # zero gradient
        
        # forward pass
        anc_out, pos_out, neg_out = model(anc_in, pos_in, neg_in)
        loss = criterion(anc_out, pos_out, neg_out)
        
        # update epoch loss by weighted batch loss
        logs['train'][epoch] += loss.sum().item()

        # backwards pass
        loss.sum().backward()
        optimizer.step()

    logs['train'][epoch] /= len(images_train)
    print("Training loss: ", logs['train'][epoch])

    # evaluate model
    model.eval()

    with torch.no_grad():
        for anc_in, pos_in, neg_in in tqdm(valloader):
            anc_in, pos_in, neg_in = anc_in.to(device), pos_in.to(device), neg_in.to(device)

            # forward pass
            anc_out, pos_out, neg_out = model(anc_in, pos_in, neg_in)
            loss = criterion(anc_out, pos_out, neg_out)

            # update epoch loss by weighted batch loss
            logs['val'][epoch] += loss.sum().item()
            
    logs['val'][epoch] /= len(images_val)
    print("Validation loss: ", logs['val'][epoch])

    scheduler.step()

    # save best model after each epoch
    if epoch > 0:
        if logs['val'][epoch] < np.amin(logs['val'][:epoch]):
            print("saving best model")
            torch.save(model, 'siameseresnet')

    # early stopping: break training loop if val loss increases for {PATIENCE} epochs
    if epoch > PATIENCE:
        if np.sum(np.diff(logs['val'])[epoch-PATIENCE:epoch] > 0) == PATIENCE:
            print("early stopping")
            break

# -------- evaluate
model = torch.load('siameseresnet')
model.to(device)
model.eval()

submission = pd.Series(index=images_test.triplets.index)
with torch.no_grad():
    for idx, (anc_in, pos_in, neg_in) in enumerate(tqdm(testloader)):
        anc_in, pos_in, neg_in = anc_in.to(device), pos_in.to(device), neg_in.to(device)

        # forward pass
        anc_out, pos_out, neg_out = model(anc_in, pos_in, neg_in)

        # loss
        loss = criterion(anc_out, pos_out, neg_out)
        submission.loc[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE-1] = loss.cpu().numpy()

# if triplet loss is larger than margin, then image B has larger distane than image C
# and thus B is the negative image (0), and C is the positive image (1)
(submission < 1).astype(int).to_csv('submission.txt', header=None, index=False)