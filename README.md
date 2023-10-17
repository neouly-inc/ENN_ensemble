# useage

## 1. build Environment (docker)

```shell
# cuda10.1 and cudnn7

# pytorch==1.6.0
# scipy==1.7.3
# scikit-learn==1.0.2
# matplotlib==3.5.3
# seaborn==0.12.2

docker build -t (name):(tag) ./
```

## 2. Dataset Annotation

### Dataset directory tree structure
```shell
./dataset/
├── ENNset
├── ENNset_each
├── subimageset  # The detailed structure is explained in the next section.
└── imageset
    ├── annotations
        ├── train.json
        ├── val.json
        └── test.json
    ├── test
        └── (image files)
    ├── train
        └── (image files)
    └── val
        └── (image files)
```



### Summary of json structure for UP-DETR dataset. It is described in the form of (field name):(type).

```json
{
    "images": [
        {
            "file_name": string,
            "height": integer,
            "width": integer,
            "id": integer  // Matches annotations["image_id"].
        },
        ...
    ],
    "annotations": [
        {
            "area": float,
            "iscrowd": integer,
            "image_id": integer,  // Matches images["id"].
            "bbox": [
                float, // Smallest x-coordinate of the box.
                float, // Smallest y-coordinate of the box.
                float, // Bounding box width.
                float  // Bounding box height.
            ],
            "category_id": integer,
            "id": integer
        },
        ...
    ],
    "categories": [  // About the whole class.
        {
            "id": integer,  // Matches annotations["category_id"].
            "name": string
        },
        ...
    ]
}
```
## other Pytorch official image classification model data summary
```shell
./dataset/
└── subimageset
    ├── train
        ├── CLASS 0             # e.g. 3004925580000
            └── (image files)
        ├── CLASS 1
            └── (image files)
        ...
    ├── val
        ├── CLASS 0
            └── (image files)
        ├── CLASS 1
            └── (image files)
        ...
    ├── test
        ├── CLASS 0
            └── (image files)
        ├── CLASS 1
            └── (image files)
        ...
```



## 3. backbone train
```shell
# Training of UP-DETR model was conducted by following the following link(https://github.com/dddzg/up-detr).
# Except UP-DETR, other models were trained in the following method.


python (root_path)/backbone/ImageClassificationModel.py
```


## 4. backbone evaluation
```shell
python backbone_evaluation.py -m (model_name)
    or
python backbone_evaluation.py -m (model_name)+(model_name)+...
```
model_name : model initial + class number + group name (e.g. R10A)

save result txt file in "(root_path)/result/backbone_eval/model_name.txt"


## 5. Generate ENN dataset
```shell 
python Generate_ENN_Dataset_each_backbone.py -d (dataset) -l (bool) -m (model_name)
    or
python Generate_ENN_Dataset_each_backbone.py -d (dataset) -l (bool) -m (model_name)+(model_name)+...
```
Please refer to the script comments for detailed explanation.

## 6. Combination ENN dataset
```shell
python Combination_ENN_Dataset.py -m (model_name) -d (dataset) -s (Class start index) -e (Class end index)
```
Please refer to the script comments for detailed explanation.

## 7. EnsembleNN Train and Total Evaluation
```shell
python (root_path)/EnsembleNN/run.py
```

