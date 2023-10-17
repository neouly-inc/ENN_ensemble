#-*- coding:utf-8 -*-

import os
import sys
import time
import random
import json

import torch
import torchvision



class Image_Classification_Model(object):
    """
    load pretrained image classification model(based on torchvision) and train the model.
    """
    def __init__(self, args: dict) -> None:
        self.seed        = args['seed']
        self.logs        = args['logs']

        self.model_name  = args['model'].lower()

        self.if_transfer = args['if_transfer']
        self.epochs      = args['epochs']
        self.batch       = args['batch']
        self.lr          = args['lr']
        self.device      = args['device'].lower()
        

        self.n_classes   = args['n_classes']
        self.img_size    = args['img_size']

        self.data_path   = args['data_path']
        self.save_path   = os.path.join(args['save_path'], self.model_name, self.model_name + "_" + args['tag'])


    def load_model(self) -> None:
        try:
            exec('self.model = torchvision.models.{}(pretrained = {})'.format(self.model_name, self.if_transfer))
            if sum([1 for i in ['resnet', 'regnet', 'googlenet', 'inception', 'shufflenet'] if i in self.model_name]) > 0:
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.n_classes)
            elif sum([1 for i in ['vgg', 'alexnet', 'mnasnet', 'mobilenet', 'convnext', 'efficientnet'] if i in self.model_name]) > 0:
                self.model.classifier[-1] = torch.nn.Linear(self.model.classifier[-1].in_features, self.n_classes)
            elif sum([1 for i in ['densenet'] if i in self.model_name]) > 0:
                self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, self.n_classes)
            elif sum([1 for i in ['squeezenet'] if i in self.model_name]) > 0:
                self.model.classifier[1] = torch.nn.Conv2d(self.model.classifier[1].in_channels, self.n_classes, self.model.classifier[1].kernel_size, self.model.classifier[1].stride)
            else:
                raise Exception("ModelNameError: Only Support ['resnet', 'regnet', 'googlenet', 'inception', 'shufflenet', 'vgg', 'alexnet', 'mnasnet', 'mobilenet', 'convnext', 'efficientnet', 'densenet', 'squeezenet']")
        except:
            raise Exception("ModelNameError: Only Support ['resnet', 'regnet', 'googlenet', 'inception', 'shufflenet', 'vgg', 'alexnet', 'mnasnet', 'mobilenet', 'convnext', 'efficientnet', 'densenet', 'squeezenet']")
        self.model.to(self.device)


    def get_dataloader(self, _type: str = 'test') -> torch.utils.data.DataLoader:
        if 'train' in str(_type).lower():
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(self.img_size[-1]),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.ColorJitter(brightness=(0.2, 2), contrast=(0.8, 1.2), saturation=(0.5, 1.5)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                torchvision.transforms.RandomErasing(p=0.3, scale=(0.02, 0.16), ratio=(0.3, 1.6), value="random"),
            ])

        elif 'val' in str(_type).lower() or 'test' in str(_type).lower():
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.img_size[-1]),
                torchvision.transforms.CenterCrop(self.img_size[-1]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            raise Exception("DataTypeError: {} is not in ['train', 'val', 'test']".format(_type))

        data_path = os.path.join(self.data_path, _type)
        if not os.path.isdir(data_path):
            raise Exception("DataPathError: {} is not directory.".format(data_path))
        label_path = self.save_path + "_" + "labels.txt"

        # ImageFolder Format
        dataset = torchvision.datasets.ImageFolder(data_path, transforms)
        if 'train' in str(_type).lower():
            self.classes = dataset.classes
            if not os.path.isfile(label_path):
                with open(label_path, "w") as file:
                    for i in self.classes:
                        file.write(i+"\n")
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch, shuffle=True, num_workers=2)
        else:
            if not os.path.isfile(label_path):
                raise Exception("DataLabelPathError: There is no file {}.".format(label_path))
            with open(label_path, 'r') as file:
                self.classes = [i for i in file]
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch, shuffle=False, num_workers=2)

        return dataloader


    def test(self, dataloader):
        test_time = time.time()
        self.model.eval()
        test_loss, test_acc1, test_acc3 = 0, 0, 0

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                pred = output.argmax(dim=1)
                loss = self.criterion(output, target)

                test_loss += loss.item()
                test_acc1 += pred.eq(target.view_as(pred)).sum().item()
                test_acc3 += (torch.topk(output, k=self.top_n, dim=1).indices == target.unsqueeze(1)).sum().item()

        test_loss /= len(dataloader.dataset)
        test_acc1 = test_acc1 / len(dataloader.dataset) * 100
        test_acc3 = test_acc3 / len(dataloader.dataset) * 100
        test_time = time.time() - test_time

        return test_time, test_loss, test_acc1, test_acc3


    def train(self, dataloader):
        train_time = time.time()
        self.model.train()
        train_loss, train_acc1, train_acc3 = 0, 0, 0

        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(data)
            pred = output.argmax(dim=1)
            loss = self.criterion(output, target)

            train_loss += loss.item()
            train_acc1 += pred.eq(target.view_as(pred)).sum().item()
            train_acc3 += (torch.topk(output, k=self.top_n, dim=1).indices == target.unsqueeze(1)).sum().item()

            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

        train_loss /= len(dataloader.dataset)
        train_acc1 = train_acc1 / len(dataloader.dataset) * 100
        train_acc3 = train_acc3 / len(dataloader.dataset) * 100
        train_time = time.time() - train_time

        return train_time, train_loss, train_acc1, train_acc3


    def run(self) -> None:
        if self.logs == 2:
            self.log_path = open(self.save_path + "_" + 'log.txt', 'w')
            sys.stdout = self.log_path

        if self.seed:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        os.makedirs(self.save_path, exist_ok=True)

        self.device = torch.device(self.device if 'cuda' in self.device and torch.cuda.is_available() else 'cpu')


        # Load Data
        if self.logs:
            load_data_time = time.time()
            print(" Data Load Start! ".center(100, "*"), flush=True)

        train_dataloader = self.get_dataloader(_type = 'train')
        val_dataloader   = self.get_dataloader(_type =   'val')
        test_dataloader  = self.get_dataloader(_type =  'test')
        if not self.n_classes:
            self.n_classes = len(self.classes)
        self.top_n = min(3, self.n_classes)

        if self.logs:
            print(" Data Load Done! Required Time: {:.4f}s ".format(time.time() - load_data_time).center(100, "*"), flush=True)
            print()


        # Load Model
        if self.logs:
            load_model_time = time.time()
            print(" {}Model Load Start! ".format("Pre-Trained " if self.if_transfer else "").center(100, "*"), flush=True)

        self.load_model()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=self.lr*0.01, last_epoch=-1)

        if self.logs:
            print(self.model)
            print(" {}Model Load Done! Required Time: {:.4f}s ".format("Pre-Trained " if self.if_transfer else "", time.time() - load_model_time).center(100, "*"), flush=True)


        # Train Model
        if self.logs:
            train_model_time = time.time()
            print(" Model Train Start! ".center(100, "*"), flush=True)

        best_epoch, best_val_loss = 0, 1e+9
        for epoch in range(1, self.epochs + 1):
            train_time, train_loss, train_acc1, train_acc3 = self.train(train_dataloader)
            val_time,   val_loss,   val_acc1,   val_acc3   = self.test(   val_dataloader)
            # test_time,  test_loss,  test_acc1,  test_acc3  = self.test(  test_dataloader)

            if self.logs:
                print('Epoch : {:4d}'.format(epoch))
                print('    Train    Time: {:.4f}, Average Loss: {:.4f}, Accuracy1: {:.4f}, Accuracy3: {:.4f}'.format(train_time, train_loss, train_acc1, train_acc3), flush=True)
                print('    Val      Time: {:.4f}, Average Loss: {:.4f}, Accuracy1: {:.4f}, Accuracy3: {:.4f}'.format(  val_time,   val_loss,   val_acc1,   val_acc3), flush=True)
                # print('    Test     Time: {:.4f}, Average Loss: {:.4f}, Accuracy1: {:.4f}, Accuracy3: {:.4f}'.format( test_time,  test_loss,  test_acc1,  test_acc3), flush=True)

            if val_loss < best_val_loss:
                best_epoch, best_val_loss = epoch, val_loss

                torch.save(self.model, self.save_path + "_" + 'best_model.pth')
                if self.logs:
                    print('Best Model Updated..!', flush=True)
            else:
                if self.logs:
                    print('Best Model Epoch: {:4d}, Val Average Loss: {:.4f}'.format(best_epoch, best_val_loss), flush=True)

            torch.save(self.model, self.save_path + "_" + 'last_model.pth')

        if self.logs:
            print(' Model Train Done! Best Epoch: {:4d}, Required Time: {:.4f}s '.format(best_epoch, time.time() - train_model_time).center(100,"*"), flush=True)
            print()


        # Eval Model
        if self.logs:
            eval_model_time = time.time()
            print(" Model Eval Start! ".center(100,"*"), flush=True)

        test_time,  test_loss,  test_acc1,  test_acc3  = self.test(  test_dataloader)

        if self.logs:
            print('    Test     Time: {:.4f}, Average Loss: {:.4f}, Accuracy1: {:.4f}, Accuracy3: {:.4f}'.format( test_time,  test_loss,  test_acc1,  test_acc3), flush=True)
            print(' Model Eval Done! Required Time: {:.4f}s '.format(time.time() - eval_model_time).center(100,"*"), flush=True)


        if self.logs == 2:
            sys.stdout = sys.__stdout__
            self.log_path.close()

        return





if __name__ == "__main__" :
    args = {
        "seed" : 42,
        "logs" : 2,    # int(0,1,2) 0 is silent, 1 is print, 2 is save to file.

        "model" : 'resnet101',  # update to torchvision==0.11.0, and then you can use efficientnet_b7 model
        "tag" : 'tag',
        "if_transfer" : True,
        "epochs" : 200, # 200+ recommanded, 3 is training test
        "batch" : 32,    # 32 recommended (resnet model, based on vram 8gb)
        "lr" : 0.005,
        "device" : 'cuda',

        "n_classes" : False,    # False is auto setting based on dataset
        "img_size" : (3, 224, 224),    # image size must be (3, 224, 224)

        "data_path" : '/workspace/IC_Data',
        "save_path" : '/workspace/backbone',
        }

    IC_Model = Image_Classification_Model(args)
    IC_Model.run()
