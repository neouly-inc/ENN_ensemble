import os
import json
import argparse
from PIL import Image
from tqdm import tqdm

from backbone.Torchvision_ImageClassificationModel import *

"""
This script that evaluates the backbone model.

Argument useage example: -m R10A      (If you want to evaluate a single model.)
                         -m R10A+R10B (If you want to evaluate a multiple model.)

The evaluation results are saved in "(root dir)/result/backbone_eval/" and output to stdout.
"""



skip_num = {"84A" : [0, 83],  "42A" : [0, 41],  "42B" : [42, 83], \
            "21A" : [0, 20],  "21B" : [21, 41], "21C" : [42, 62], "21D" : [63, 83], \
            "10A" : [0, 10],  "10B" : [11, 20], "10C" : [21, 31], "10D" : [32, 41], "10E" : [42, 52], \
            "10F" : [53, 62], "10G" : [63, 73], "10H" : [74, 83]}

Torchvision_model_dic = {"R": "resnet101", "W": "wide_resnet101_2"}


def load_data_path():
    return json.load(open(os.path.join(root_dir, "dataset", "imageset", "annotations", "test.json")))


def load_labels(switch):
    with open(os.path.join(root_dir, "backbone", Torchvision_model_dic[switch[0]], Torchvision_model_dic[switch[0]] + "_" + switch[1:4] + "_labels.txt")) as names:
        classes = names.read().split("\n")
    with open(os.path.join(root_dir,"labels", "total_labels.txt")) as names:
        total_classes = names.read().split("\n")
    names_dic, each_acc_dic = {}, {}

    for i, cla in enumerate(classes):
        names_dic[i] = classes[i]
        each_acc_dic[cla] = [0, 0, total_classes[i + skip_num[switch[1:4]][0]]]
    return names_dic, each_acc_dic


def load_model(root_dir, switch):
    _path = os.path.join(root_dir, "backbone")
    return Torchvision_ImageClassificationModel_Ready(_path, mode = Torchvision_model_dic[switch[0]])


def Evaluation(root_dir, switch):
    data = load_data_path()
    names_dic, each_acc_dic = load_labels(switch)
    TestModel = load_model(root_dir, switch)

    path = os.path.join(root_dir, "dataset", "imageset", "test")
    total, t1_correct, t3_correct = 0, 0, 0
    for ind, meta in tqdm(enumerate(data["images"])):
        # load image and label
        label = data["annotations"][ind]["category_id"] - 1
        if label > skip_num[switch[1:4]][1] or label < skip_num[switch[1:4]][0]:
            continue
        else:
            true_num = label - skip_num[switch[1:4]][0]
        total += 1

        pil_img = Image.open(os.path.join(path, meta["file_name"]))

        # predict(serial)
        pred = TestModel.Top_N(pil_img, switch[1:4])

        res = sorted(pred.items(), key=(lambda x:x[1]), reverse = True)
        top3_pred_name =[str(res[0][0]), str(res[1][0]), str(res[2][0])]

        # calculate
        true_name = names_dic[true_num]
        each_acc_dic[true_name][1] += 1
        if true_name == top3_pred_name[0]:
            t1_correct += 1
            each_acc_dic[true_name][0] += 1

        if true_name in top3_pred_name:
            t3_correct += 1

    # print and save result
    print("Accuracy :", round(t1_correct / total, 5), round(t3_correct / total, 5))
    strFormat = '%-40s%-8s%-5s\n'
    strOut = strFormat % ('names','correct','total')
    os.makedirs(os.path.join(root_dir, "result", "backbone_eval"), exist_ok=True)
    with open(os.path.join(root_dir, "result", "backbone_eval", switch + ".txt"), "w") as f:
        f.write("Accuracy :" + "  " + str(round(t1_correct / total, 5)) + "  " + str(round(t3_correct / total, 5)) + "\n")
        for i in each_acc_dic:
            strOut += strFormat %(each_acc_dic[i][2], each_acc_dic[i][0], each_acc_dic[i][1])
            f.write(str(each_acc_dic[i][2]) + "\t" + str(each_acc_dic[i][0]) + " " + str(each_acc_dic[i][1]) + "\n")
    print(strOut)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default="R84A", help='Select evaluation backbone model. For usage instructions, refer to the main comment of the script. [defualt : U84A]')
    args = parser.parse_args()

    root_dir, _ = os.path.split(os.path.abspath(__file__))

    for switch in args.mode.split("+"):
        Evaluation(root_dir, switch)