#-*- coding:utf-8 -*-

import os
import time
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import *
from dataloader import *
from utils import *





def main(args: dict) -> None:
    """
    This Function Training and Evaluating EnsembleNN.

    Input
        args   = Input to Training and Assessment (dict)
        * For more Information about args, See the run Section at the bottom.
    """
    seed          = args['seed']
    batch_size    = args['batch_size']
    model_tag     = args['tag']
    save_path     = os.path.join(args['save_path'], model_tag)
    data_path     = os.path.join(args['data_path'], model_tag)
    cat_info      = args['cat_info']
    input_size    = sum(cat_info)    # num of categories
    output_size   = len(cat_info)    # num of sub models
    use_cuda      = args['use_cuda']
    epochs        = args['epochs']
    learning_rate = args['learning_rate']
    logs          = args['logs']
    top_k         = min(3, input_size)

    torch.manual_seed(seed)
    os.makedirs(save_path, exist_ok=True)

    if logs==2:
        log_path = open(os.path.join(save_path, 'log.txt'), 'w')
        sys.stdout = log_path


    if logs:
        print(" Model Load Start! ".center(100,"*"), flush=True)
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    
    model = EnsembleNN(in_size=input_size, out_size=output_size, drop_rate=0.2)
    if logs:
        print(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    if logs:
        print(" Model Load Done! ".center(100,"*"), flush=True)
        print()

    if logs:
        print(" Data Load Start! ".center(100,"*"), flush=True)
    train_dataloader, val_dataloader, test_dataloader = Load_Data(data_path, cat_info, batch_size, logs, seed)
    if logs:
        print(" Data Load Done! ".center(100,"*"), flush=True)
        print()
    

    if logs:
        print(" Model Train Start! ".center(100,"*"), flush=True)
    required_time = time.time()
    best_epoch, best_val_loss = 0, 1e+9
    for epoch in range(1, epochs + 1):
        model, train_time, train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
        val_time,   val_loss,  val_acc = test(model,  val_dataloader, criterion, device)
        # test_time, test_loss, test_acc = test(model, test_dataloader, criterion, device)

        if logs:
            print('Epoch : {:4d}'.format(epoch))
            print('    Train    Time: {:.4f}, Average Loss: {:.4f}, Accuracy: {:.4f}'.format(train_time, train_loss, train_acc), flush=True)
            print('    Val      Time: {:.4f}, Average Loss: {:.4f}, Accuracy: {:.4f}'.format(  val_time,   val_loss,   val_acc), flush=True)
            # print('    Test     Time: {:.4f}, Average Loss: {:.4f}, Accuracy: {:.4f}'.format( test_time,  test_loss,  test_acc), flush=True)

        if val_loss < best_val_loss:
            best_epoch, best_val_loss = epoch, val_loss

            torch.save(model, os.path.join(save_path, 'best_model.pth'))
            if logs:
                print('Best Model Updated..!', flush=True)
        else:
            if logs:
                print('Best Model Epoch: {:4d}, Val Average Loss: {:.4f}'.format(best_epoch, best_val_loss), flush=True)

        torch.save(model, os.path.join(save_path, 'last_model.pth'))

    if logs:
        print(' Model Train Done! Best Epoch: {:4d}, Required Time: {:.4f}s '.format(best_epoch, time.time() - required_time).center(100,"*"), flush=True)
        print()


    if logs:
        print(" Model Eval Start! ".center(100,"*"), flush=True)
    eval_time, eval_acc, eval_raw_acc, eval_raw_acc3, conf_real, conf_pred = eval(save_path, cat_info, test_dataloader, top_k, device)
    conf_matrix(save_path, conf_real, conf_pred)


    if logs:
        print('Best Eval    Time: {:.4f}, DNN Accuracy: {:.4f}, ENN Accuracy: {:.4f}, ENN Accuracy(Top 3): {:.4f}'.format(eval_time, eval_acc, eval_raw_acc, eval_raw_acc3), flush=True)
        print(" Model Eval Done! ".center(100,"*"), flush=True)

    if logs==2:
        sys.stdout = sys.__stdout__
        log_path.close()

    return





if __name__ == "__main__":
    args = dict()

    # for All
    args['seed']          = 42                               # int(42)      :  Random Seed.
    args['batch_size']    = 256                              # int(256)     :  Depending on the Size of Using Data.
    args['tag']           = "tag"                            # str          :  Model Tag.
    args['save_path']     = "/workspace/EnsembleNN_output"   # str          :  Directory Path where Models and Logs will be Stored.

    # for Data
    args['data_path']     = "/workspace/dataset/ENNset"      # str          :  Directory Path where the Data is stored.
    args['cat_info']      = [11, 10, 11, 10, 11, 10, 11, 10] # list         :  Number of Categories for each Sub Model (list[sub_model_1_cat_len, sub_model_2_cat_len, ..., sub_model_N_cat_len]).

    # for Model
    args['use_cuda']      = True                             # bool(True)   :  True if you want to use GPU calculations, False otherwise.

    # for Train
    args['epochs']        = 200                              # int(200)     : about 200 or higher is recommended.
    args['learning_rate'] = 0.005                            # float(0.005) : about 0.005 is recommended.
    args['logs']          = 2                                # int(0,1,2)   :  0 is Silent, 1 is Print, 2 is Save to File.

    # Run All
    main(args)

