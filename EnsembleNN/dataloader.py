#-*- coding:utf-8 -*-

import os
import pickle as pkl


import numpy as np
import torch





def Load_Data(data_path: str, cat_info: list, batch_size: int = 256, logs: int = 0, seed: int = 42, num_workers: int = 2):
    """
    This Function Generates Data Randomly.
    Please Check to the Code Below and Change it.

    Input
        data_path   = Stoed Sun Model Data Path (str)
        cat_info    = Number of Categories for each Sub Model (list[sub_model_1_cat_len, sub_model_2_cat_len, ..., sub_model_N_cat_len])
        batch_size  = DataLoader Batch Size (int, Affected by th size of GPU VRAM)
        logs        = 0 is Silent, 1 is Print, 2 is Save to File. (int[0, 1, 2], Default = 0)
        seed        = Set Torch Random Seed (int, Default = 42)
        num_workers = DataLoader Worker Size (int, Default = 2, Affected by the number of CPU Cores)

    Output
        train dataloader = torch.utils.data.DataLoader
        val dataloader   = torch.utils.data.DataLoader
        test dataloader  = torch.utils.data.DataLoader
    """
    torch.manual_seed(seed = seed)

    # # ============================== Data Generation for Operational Verification ==============================
    # in_size = sum(cat_info)
    # train_size, val_size, test_size = [60000, 20000, 20000]
    # eye = torch.eye(in_size, dtype=torch.float32)
    # train_x = torch.randn(size=(train_size, in_size), dtype=torch.float32)
    # val_x   = torch.randn(size=(  val_size, in_size), dtype=torch.float32)
    # test_x  = torch.randn(size=( test_size, in_size), dtype=torch.float32)
    # train_y = torch.stack([eye[i] for i in (train_x.argmax(dim=1))]).to(torch.long)
    # val_y   = torch.stack([eye[i] for i in (  val_x.argmax(dim=1))]).to(torch.long)
    # test_y  = torch.stack([eye[i] for i in ( test_x.argmax(dim=1))]).to(torch.long)
    # if logs:
    #     print("Data Generated!")
    # # ============================== Data Generation for Operational Verification ==============================

    with open(os.path.join(data_path, "train.pkl"), 'rb') as f:
        train_x, train_y = pkl.load(f)
    with open(os.path.join(data_path,   "val.pkl"), 'rb') as f:
        val_x,     val_y = pkl.load(f)
    with open(os.path.join(data_path,  "test.pkl"), 'rb') as f:
        test_x,   test_y = pkl.load(f)

    train_x, train_y = torch.from_numpy(np.array(train_x)).to(torch.float32), torch.from_numpy(np.array(train_y)).to(torch.long)
    val_x,     val_y = torch.from_numpy(np.array(  val_x)).to(torch.float32), torch.from_numpy(np.array(  val_y)).to(torch.long)
    test_x,   test_y = torch.from_numpy(np.array( test_x)).to(torch.float32), torch.from_numpy(np.array( test_y)).to(torch.long)

    if logs:
        print("Train_X    : {},  Val_X    : {},  TEST_X    : {}".format(train_x.shape, val_x.shape, test_x.shape))
        print("Train_Y    : {},  Val_Y    : {},  TEST_Y    : {}".format(train_y.shape, val_y.shape, test_y.shape))

    src = dst = 0
    ens_train_y, ens_val_y, ens_test_y = [], [], []
    for v in cat_info:
        src, dst = dst, dst + v
        ens_train_y.append(train_y[:, src : dst].sum(dim=1))
        ens_val_y.append(    val_y[:, src : dst].sum(dim=1))
        ens_test_y.append(  test_y[:, src : dst].sum(dim=1))
    ens_train_y = torch.stack(ens_train_y, dim=1)
    ens_val_y   = torch.stack(  ens_val_y, dim=1)
    ens_test_y  = torch.stack( ens_test_y, dim=1)

    if logs:
        print("Ens_Train_Y: {},  Ens_Val_Y: {},  Ens_TEST_Y: {}".format(ens_train_y.shape, ens_val_y.shape, ens_test_y.shape))

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y, ens_train_y)
    val_dataset   = torch.utils.data.TensorDataset(  val_x,   val_y,   ens_val_y)
    test_dataset  = torch.utils.data.TensorDataset( test_x,  test_y,  ens_test_y)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers)
    val_dataloader   = torch.utils.data.DataLoader(  val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers)
    test_dataloader  = torch.utils.data.DataLoader( test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


