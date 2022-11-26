# Contour Transformer Network
Code for the paper "Learning to segment anatomical structures accurately from one exemplar" (MICCAI 2020) and "Contour transformer network for one-shot segmentation of anatomical structures" (TMI 2020).

## Dataset
Please find our datasets in this <a href="https://github.com/rudylyh/CTN_data" target="_blank">repo</a>.

## How to use
Take the knuckle data as example.
+ ##### One-shot
  1. **Training**:
    + **Input**:
      + A folder of knuckle data. Each subfolder contains three files: (1) IMG_NAME.png (the knuckle image); (2) IMG_NAME_init.json (the init contour); (3) IMG_NAME_gt.json (the ground truth contour). Specify the one-shot sample in 'Experiments/knuckle-contour.json'. Only the ground truth contour of the one-shot sample is used in this step.
      + A list of image names for training and testing ('cvpr/example_dataset.json').
    + **Output**: A checkpoint folder ('Checkpoints/knuckle_11_06_19_35_18'). The suffix numbers refer to the start time of training.
    + **Scripts**:
      + `python Scripts/train/train_contour.py --exp Experiments/knuckle-contour.json`
      + (For lung) `python Scripts/train/train_contour.py --exp Experiments/knee-contour.json` (Modify the json file to train models for femur and tibia, respectively.)
      + (For knee) `python Scripts/train/train_contour.py --exp Experiments/knuckle-contour.json` (Modify the json file to train models for left and right lungs, respectively.)
  2. **Validation**
    + `python Scripts/train/eval_contour.py --exp Experiments/knuckle-contour.json --resume Checkpoints/knuckle_11_06_19_35_18/best.pth`

+ ##### Partial-supervised (Human in the loop)
  1. **Predict contours for all training images**
    + **Input:**
      + The above knuckle data folder.
      + A one-shot model (Checkpoints/knuckle_11_06_19_35_18/best.pth).
    + **Output:**
      + A json file of predicted contour in each image folder (IMG_NAME_pred_knuckle_11_06_19_35_18.json).
      + A list of image names sorted by the Hausdorff distance between the predicted contour and the ground truth contour in descending order. ('Checkpoints/knuckle_11_06_19_35_18/train_names_sorted_by_dist.json')
    + **Scripts:**
        + Modify 'Scripts/train/eval_contour.py' to run the function generate_pred_for_train().
        + `python Scripts/train/eval_contour.py --exp Experiments/knuckle-contour.json --resume Checkpoints/knuckle_11_06_19_35_18/best.pth`
  2. **Finetune**
      + Modify 'Experiments/knuckle-contour.json'. Change 'use_partial_sup' to true. Specify 'sample_percent' (How many training images are partial-supervised).
      + `python Scripts/train/train_contour.py --exp Experiments/knuckle-contour.json --resume Checkpoints/knuckle_11_06_19_35_18/best.pth`
      + Output another checkpoint folder ('Checkpoints/knuckle_11_08_17_12_32').

## Citation
```
@inproceedings{lu2020learning,
  title={Learning to segment anatomical structures accurately from one exemplar},
  author={Lu, Yuhang and Li, Weijian and Zheng, Kang and Wang, Yirui and Harrison, Adam P and Lin, Chihung and Wang, Song and Xiao, Jing and Lu, Le and Kuo, Chang-Fu and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={678--688},
  year={2020},
  organization={Springer}
}
@article{lu2020contour,
  title={Contour transformer network for one-shot segmentation of anatomical structures},
  author={Lu, Yuhang and Zheng, Kang and Li, Weijian and Wang, Yirui and Harrison, Adam P and Lin, Chihung and Wang, Song and Xiao, Jing and Lu, Le and Kuo, Chang-Fu and others},
  journal={IEEE transactions on medical imaging},
  volume={40},
  number={10},
  pages={2672--2684},
  year={2020},
  publisher={IEEE}
}
```

## Acknowledgement
This repo is partially based on <a href="https://github.com/fidler-lab/curve-gcn" target="_blank">Curve-GCN</a>.
