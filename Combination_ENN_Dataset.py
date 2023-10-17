import os, sys
import pickle
import argparse

"""
This script combines each created enn dataset into one.

Argument useage example: -m R10A+R10B (If you want to evaluate a multiple model.)
                         -d test    (all, train, val and test)
                         -s 0       (Starting index corresponding to the current model among all classes)
                         -e 20      (Last index corresponding to the current model among all classes)

The combined input results are saved in "(root dir)/dataset/ENNset/(model_name)".
"""




def load_label(root_dir, mode):
    # load
    return pickle.load(open(os.path.join(root_dir, "dataset", "ENNset_each", "labels", mode + ".pkl"), "rb"))


def load_inputs(model_name, mode):
    models = model_name.split("+")
    for ind, model in enumerate(models):
        input_values = pickle.load(open(os.path.join(root_dir, "dataset", "ENNset_each", model, mode + ".pkl"), "rb"))
        if ind == 0:
            concat_all = input_values
        else:
            for indx, feature in enumerate(concat_all):
                feature.extend(input_values[indx])
                concat_all[indx] = feature
    return concat_all


def Combination_dataset(root_dir, model_name, mode, class_start, class_end):
    print(model_name, mode, class_start, class_end)
    labels = load_label(root_dir, mode)
    concat_all_inputs = load_inputs(model_name, mode)

    x_train = []
    y_train = []
    for ind, label in enumerate(labels):
        if label.index(1) >= class_start and label.index(1) <= class_end:
            y_train.append(label[class_start : class_end + 1])
            x_train.append(concat_all_inputs[ind])

    print("Shape of Train X :", (len(x_train), len(x_train[0])))
    print("Shape of Train Y :", (len(y_train), len(y_train[0])))

    os.makedirs(os.path.join(root_dir, "dataset", "ENNset", model_name), exist_ok=True)

    with open(os.path.join(root_dir, "dataset", "ENNset", model_name, mode + ".pkl"), "wb") as f:
        pickle.dump((x_train, y_train), f)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default="R10A+R10B", help='Select evaluation backbone model. For usage instructions, refer to the main comment of the script. [defualt : U84A]')
    parser.add_argument('-d', '--data', type=str, default="all", help='Select dataset. i.e. all, train, val and test [defualt : all]')
    parser.add_argument('-s', '--class_start', type=int, default=0, help='Class start index. [defualt : 0]')
    parser.add_argument('-e', '--class_end', type=int, default=83, help='Class end index. [defualt : 83]')
    args = parser.parse_args()
    root_dir, _ = os.path.split(os.path.abspath(__file__))

    Combination_dataset(root_dir, args.mode, args.data, args.class_start, args.class_end)