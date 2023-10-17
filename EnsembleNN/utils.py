#-*- coding:utf-8 -*-

import os
import sys
import time

import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import *





def train(model, dataloader, optimizer, criterion, device):
    """
    This Function Train the EnsembleNN Model per each Epoch.

    Input
        model       = PyTorch Model (torch.nn.Module)
        dataloader  = Train Dataloader (torch.utils.data.DataLoader)
        optimizer   = torch.optim.AdamW
        criterion   = Loss Function (torch.nn.CrossEntropyLoss)
        device      = Location of allocation to operate on (torch.device)
    Output
        model       = Trained PyTorch Model (torch.nn.Module)
        train_time  = Seconds used for Training per each Epoch (float)
        train_loss  = Training Average Loss per each Epoch (float)
        train_acc   = Training Total Accuracy Percentage per each Epoch (float)
    """
    train_time = time.time()
    train_acc, train_loss = 0, 0

    model.train()
    for data, _, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        real = target.argmax(dim=1)
        pred = output.argmax(dim=1)
        loss = criterion(output, real)

        train_loss += loss.item()
        train_acc += pred.eq(real.view_as(pred)).sum().item()

        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader.dataset)
    train_acc = train_acc / len(dataloader.dataset) * 100
    train_time = time.time() - train_time

    return model, train_time, train_loss, train_acc


def test(model, dataloader, criterion, device):
    """
    This Function Test the EnsembleNN Model per each Epoch.

    Input
        model       = Trained PyTorch Model (torch.nn.Module)
        dataloader  = Test(Validation) Dataloader (torch.utils.data.DataLoader)
        criterion   = Loss Function (torch.nn.CrossEntropyLoss)
        device      = Location of allocation to operate on (torch.device)
    Output
        test_time   = Seconds used for Testing per each Epoch (float)
        test_loss   = Testing Average Loss per each Epoch (float)
        test_acc    = Testing Total Accuracy Percentage per each Epoch (float)
    """
    test_time = time.time()
    test_acc, test_loss = 0, 0

    model.eval()
    with torch.no_grad():
        for data, _, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            real = target.argmax(dim=1)
            pred = output.argmax(dim=1)
            loss = criterion(output, real)

            test_loss += loss.item()
            test_acc += pred.eq(real.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)
    test_acc = test_acc / len(dataloader.dataset) * 100
    test_time = time.time() - test_time

    return test_time, test_loss, test_acc


def eval(save_path, cat_info, dataloader, top_k, device) -> tuple:
    """
    This Function Final Evalutae the EnsembleNN Model per each Epoch.

    Input
        save_path     = Best Model Path for Trained PyTorch Model (str)
        cat_info      = Number of Categories for each Sub Model (list[sub_model_1_cat_len, sub_model_2_cat_len, ..., sub_model_N_cat_len])
        dataloader    = Test Dataloader (torch.utils.data.DataLoader)
        top_k         = top_k k to measure accuracy, default 3 (int)
        device        = Location of allocation to operate on (torch.device)
    Output
        eval_time     = Seconds used for Evaluation per each Epoch (float)
        eval_acc      = Testing EnsembleNN Accuracy Percentage per each Epoch (float)
        eval_raw_acc  = Testing Final EnsembleNN Accuracy Percentage per each Epoch (float)
        eval_raw_acc3 = Testing Final EnsembleNN Top 3 Accuracy Percentage per each Epoch (float)
        conf_real     = Final EnsembleNN Target(Real) Value List needed to draw a Confusion Matrix (list)
        conf_pred     = Final EnsembleNN Predicted Value List needed to draw a Confusion Matrix (list)
    """
    eval_time = time.time()
    OF = OverlayFunc(cat_info)
    eval_acc, eval_raw_acc, eval_raw_acc3 = 0, 0, 0
    conf_real, conf_pred = [], []

    model = torch.load(os.path.join(save_path, 'best_model.pth')).to(device)
    model.eval()

    with torch.no_grad():
        for data, raw_target, target in dataloader:
            data, raw_target, target = data.to(device), raw_target.to(device), target.to(device)

            output = model(data)
            real = target.argmax(dim=1, keepdim=True)
            pred = output.argmax(dim=1, keepdim=True)
            raw_real = raw_target.argmax(dim=1, keepdim=True)
            final = OF.run(data, output.softmax(dim=1))    # Use Softmax for Data Range
            
            top_3_real = torch.cat([raw_real for _ in range(top_k)],dim=1)
            top_3_final = final.topk(k=top_k, dim=1).indices
            final = final.argmax(dim=1, keepdim=True)
            
            conf_real.extend(raw_real.squeeze().cpu().numpy().tolist())
            conf_pred.extend(final.squeeze().cpu().numpy().tolist())

            eval_acc += pred.eq(real.view_as(pred)).sum().item()
            eval_raw_acc += final.eq(raw_real.view_as(final)).sum().item()
            eval_raw_acc3 += top_3_final.eq(top_3_real.view_as(top_3_final)).sum().item()

    
    eval_acc = eval_acc / len(dataloader.dataset) * 100
    eval_raw_acc = eval_raw_acc / len(dataloader.dataset) * 100
    eval_raw_acc3 = eval_raw_acc3 / len(dataloader.dataset) * 100
    eval_time = time.time() - eval_time

    return eval_time, eval_acc, eval_raw_acc, eval_raw_acc3, conf_real, conf_pred


def conf_matrix(save_path: str, y_real: list, y_pred: list) -> None:
    """
    This Function is draws a Confusion Matrix as the Result of Final EnsembleNN Model.

    Input
        save_path   = Where to Store the Confusion Matrix (str)
        conf_real   = Final EnsembleNN Target(Real) Value List needed to draw a Confusion Matrix (list)
        conf_pred   = Final EnsembleNN Predicted Value List needed to draw a Confusion Matrix (list)
    """
    conf_matrix = metrics.confusion_matrix(y_real, y_pred)

    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(conf_matrix, cmap=cm.gray, xticklabels=5, yticklabels=5)
    plt.title('ENN Model Confusion Matrix')
    plt.xlabel('Predicted Class Index')
    plt.ylabel('True Class Index')
    plt.savefig(os.path.join(save_path, "conf_matrix.jpg"))

    return 

