#-*- coding:utf-8 -*-

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torchvision.transforms as T
torch.set_grad_enabled(False)
import os
import sys
import numpy as np


class Torchvision_ImageClassificationModel_Ready :
    """
    Torchvision Image Classification Model Testset Code
    """
    def __init__(self, _p, mode) :
        self.path = _p
        self.mode = mode
        self.model_path = self.path + "/" + self.mode + "/"
        sys.path.append(os.path.dirname(os.path.abspath(self.model_path)))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.limit_resize = 224    # now 224 only
        self.top_n = 3
        self.transform = T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        self.switch_check = []
        self.switch_now = False
    
    def image_resizing(self, PIL_Img):    # PIL Image size to 224 x 224
        x, y = PIL_Img.size
        if x == y == self.limit_resize:
            return PIL_Img
        else :
            sz = max(x, y)
            PIL_Img = PIL_Img.resize((int(x/(sz/self.limit_resize)), int(y/(sz/self.limit_resize))), Image.LANCZOS)
            resized_img = Image.new(mode = 'RGB', size = (self.limit_resize, self.limit_resize), color = 'black')
            resized_img.paste(PIL_Img)
            return resized_img
    
    def switch_model(self, tag):    # switch model, classes, n_classes,  based on mode
        if self.switch_now == tag:
            pass
        elif tag not in self.switch_check :
            _path = self.model_path + self.mode + '_' + tag
            self.model = torch.load(_path + "_best_model.pth").to(self.device).eval()
            self.classes = [line.rstrip('\n') for line in open(_path+"_labels.txt", 'r')]
            self.n_classes = len(self.classes)
            print("Model Changed {} to {}".format(self.switch_now, tag))
            self.switch_now = tag
            self.switch_check.append(tag)
        else :
            raise Exception("Error occurred : There is no tag '{}' Check your mode again.".format(tag))
        return 0
    
    @torch.no_grad()
    def predict(self, im):    # model predict (default)
        im = self.transform(self.image_resizing(im)).unsqueeze(0).to(self.device)
        return self.model(im).softmax(-1)[0,:].detach().to('cpu').numpy()
        
    def Top_N(self, im, tag):    # Output in the form of predicted value {key(str) : value(float)}
        self.switch_model(tag)
        probas = self.predict(im)
        sorted_k = probas.argsort()[::-1]
        return {self.classes[_] : probas[_] for _ in sorted_k}
    
    def Top_3(self, im, tag) :    # ['Top1_key','Top2_key','Top3_key'], ['Top1_value','Top2_value','Top3_value']
        self.switch_model(tag)
        probas = self.predict(im)
        sorted_k = probas.argsort()[::-1][:self.top_n]
        return [self.classes[_] for _ in sorted_k], [round(probas[_],3) for _ in sorted_k]



if __name__ == "__main__" :
    # Test Model Dir Setting
    model_name = 'resnet101'
    _path = '/workspace/test/{}_model/'.format()

    # Test Model Load
    TestModel = Torchvision_ImageClassificationModel_Ready(_path, mode = model_name)
    
    # Load Image
    _img_path = '/workspace/test/testset/img.jpg'
    img = Image.open(_img_path)

    # Test Image
    tag = 'tag'    # Enter an appropriate tag.
    res1 = TestModel.Top_N(img, tag)
    print("res1: ", res1)
    res2 = TestModel.Top_3(img, tag)
    print("res2: ", res2)

    



"""
output statement

Model Changed False to tag   << This is printed only once for the first time when a tag is changed.

res1: {'class_1': prob_1, 'class_2': prob_2, ..., 'class_n': prob_n}

res2: (['class_top1', 'class_top2', 'class_top3'], [prob_top1, prob_top2, prob_top3])
"""