# DETR-tf2
This project implements the SOTA anchor-free objection detection algorithm DETR

## download MS COCO 2017

download COCO2017 dataset from [here](https://cocodataset.org/). unzip directory train2017, val2017 and annotations. generate dataset with the following command.

```bash
python3 create_datasets.py </path/to/train2017> </path/to/val2017> </path/to/annotations>
```

upon executing the script successfully, there will be directories named trainset and testset under the root directory of the source code.

## train the model

train the model by executing the following command

```bash
python3 train.py
```

save the model from checkpoint with command

```bash
python3 save_model.py
```

## known issue

currently, due to tensorflow's bug descripted [here](https://github.com/tensorflow/tensorflow/issues/46035). the training code cannot work properly. let's wait for tensorflow's perfect support of ragged tensor.
