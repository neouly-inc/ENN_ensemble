import os
from PIL import Image
from tqdm import tqdm
import pickle
import sys
import json
import argparse

from backbone.Torchvision_ImageClassificationModel import *

"""
This is a script that generates each enn input through each backbone model.

Argument useage example: -m R10A      (If you want to evaluate a single model.)
                         -m R10A+R10B (If you want to evaluate a multiple model.)
                         -d test      (all, train, val and test.)
                         -l True      (Set True if you want to create labels for the entire class.
                                       You only need to run it once for the first time.)

Each Enn input are saved in "(root dir)/dataset/ENNset_each/(model_name)".
And label is saved in "(root dir)/dataset/ENNset_each/labels".
"""



Torchvision_model_dic = {"R": "resnet101", "W": "wide_resnet101_2"}


def load_data_path(data):
    return json.load(open(os.path.join(root_dir, "dataset", "imageset", "annotations", data + ".json")))


def load_model(root_dir, switch):
    _path = os.path.join(root_dir, "backbone")
    return Torchvision_ImageClassificationModel_Ready(_path, mode = Torchvision_model_dic[switch[0]])


def load_labels():
    with open(os.path.join(root_dir, "labels", "total_labels.txt")) as names:
        total_classes = names.read().split("\n")
    names_dic = {}
    for i, cla in enumerate(total_classes):
        names_dic[total_classes[i]] = i
    return names_dic


def generate_each_ENN_dataset(root_dir, switch, data, save_label=False):
    datas = load_data_path(data)
    names_dic = load_labels()
    TestModel = load_model(root_dir, switch)

    # test start
    path = os.path.join(root_dir, "dataset", "imageset")
    dnn_res, label_res = [], []
    for img in tqdm(datas["images"]):
        # load image and label
        pil_img = Image.open(os.path.join(path, data, img["file_name"]))
        if save_label:
            for anno in datas["annotations"]:
                if img["id"] == anno["image_id"]:
                    label = anno["category_id"] - 1
                    break
        # predict(serial)
        pred = TestModel.Top_N(pil_img, switch[1:4])
        matching_table = {}
        for i, j in pred.items():
            matching_table[names_dic[i]] = (i, j)
        res = sorted(matching_table.items(), key=(lambda x:x[0]))
        dnn_input = [x[1][1] for x in res]
        dnn_res.append(dnn_input)

        if save_label:
            label_res.append([1 if ind == label else 0 for ind, _ in enumerate(range(len(names_dic)))])


    save_path = os.path.join(root_dir, "dataset", "ENNset_each", switch)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, data + ".pkl"), "wb") as f:
        pickle.dump(dnn_res, f)
    
    if save_label:
        label_save_path = os.path.join(root_dir, "dataset", "ENNset_each", "labels")
        os.makedirs(label_save_path, exist_ok=True)
        with open(os.path.join(label_save_path, data + ".pkl"), "wb") as f:
            pickle.dump(label_res, f)


    print("\n Done! {}, {}".format(switch, data))



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default="R84A", help='Select evaluation backbone model. For usage instructions, refer to the main comment of the script. [defualt : U84A]')
    parser.add_argument('-d', '--data', type=str, default="all", help='Select dataset. i.e. all, train, val and test [defualt : all]')
    parser.add_argument('-l', '--save_label', type=bool, default=False, help='Save entire label. i.e. True or False [defualt : False]')
    args = parser.parse_args()
    print(args)
    root_dir, _ = os.path.split(os.path.abspath(__file__))

    if args.data == "all":
        data = ["train", "val", "test"]
    else:
        data = [args.data]

    for ind, switch in enumerate(args.mode.split("+")):
        for set in data:
            if ind == 0:
                generate_each_ENN_dataset(root_dir, switch, set, save_label=args.save_label)
            else:
                generate_each_ENN_dataset(root_dir, switch, set, save_label=False)