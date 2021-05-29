from time import time

from model import cnn_model
from data import load_data
import math

import torch
import torch.nn as nn

from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

NUM_CLASSES = 43
IMG_SIZE = 48
lr = 0.001
argepoch = 2
batch_size = 16


torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = cnn_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.90)
criterion = nn.CrossEntropyLoss()





def train(train_loader):
    model.train()
    iter = 0
    total_loss = 0.0

    for i, (img, labels) in enumerate(train_loader):
        img, labels = img.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        iter += 1

        if iter % 200 == 0:
            cur_loss = total_loss / 200
            print('cur_loss{:5.2f}'.format(cur_loss))
            total_loss = 0.0


def evaluate(eval_model, val_loader):
    eval_model.eval()
    total_loss = 0.0
    total = 0
    corsum = 0
    with torch.no_grad():
        for i, (img, labels) in enumerate(val_loader):
            img, labels = img.to(device), labels.to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels.long())
            total_loss += loss.item()
            total += labels.size(0)
            print(predicted)
            print(labels)
            corsum += torch.eq(predicted, labels).sum().float().item()
        
    return total_loss / total, corsum / total



if __name__=='__main__':
    trainloader, valloader, testloader = load_data()
    best_loss = float("inf")
    best_model = None

    for epoch in range(1, argepoch):
        train(trainloader)
        val_loss, val_acc = evaluate(model, valloader)
        print('end of epoch {:3d} | val_loss {:5.2f} | val_acc {:5.2f}'.format(epoch, val_loss, val_acc))
    
        if val_loss < best_loss:
            print("saving model.......")
            best_loss = val_loss
            best_model = model
        
        scheduler.step()

    torch.save(best_model, 'best_model.pt')

    test_loss, test_acc = evaluate(best_model, testloader)
    print('=' * 90)
    print('| End of training | test loss {:5.2f} | test acc {:5.2f}'.format(
        test_loss, test_acc))
    print('=' * 90)
