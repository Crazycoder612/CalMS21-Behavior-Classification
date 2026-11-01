# CalMS21 Dataset

Dataset website: https://sites.google.com/view/calms21

The study of naturalistic social behavior requires quantification of animals’ interactions. This is generally done through manual annotation—a highly time consuming and tedious process. Recent advances in computer vision enable tracking the pose (posture) of freely-behaving animals. However, automated classification complex social behaviors often requires large amounts of hand-labeled training data, which can only be produced by trained domain experts.

Included in this dataset are pose estimates (with confidence scores) and frame-by-frame manual annotations of video recordings of pairs of mice freely interacting in a standard home cage. Additionally, we release pre-computed features using task programming trained on the unlabeled subset of CalMS21.

## Overview
We provide training and test sets for three behavior classification tasks, as well as a large collection of unannotated videos that can be used for unsupervised pose analysis. For best practices, only the train set should be used during development (and a validation set can be held out from the train set), and after model development, results should be reported on the test set across multiple runs with different random seeds.

Each dataset consists of **pose estimates** and **framewise behavior annotations** for a collection of top-view videos of socially interacting mice. Videos are collected in a standardized behavioral arena, and are sampled from a pool spanning *five years* of experimental data. Lighting and contrast may vary modestly from video to video, resulting in some variability in the quality of pose estimates.

![example behavior video with pose estimates](https://drive.google.com/file/d/1kmV-OqB9TZIKC7AOMQxpg2ozvl_ShD7y/view?usp=sharing)

**Training and test sets are split by animal identity**- animals in the test set are recorded in the same experimental setup as those in the training set. Experiments are sampled from the same range of dates, and annotated by the same individual who annotated the corresponding training set.

* **Behavior**: A domain expert-defined action of one or both animals, such as attack, mount, or investigation. Most of the behaviors included in this challenge are social behaviors, involving the position and movements of both animals. **Unless specifically noted, annotations refer specifically to behaviors initiated by the black mouse, the "resident" in this assay**.

## Data format

There are four zip files, one for the large unlabeled set, and three for each of the tasks (classic classification, annotation styles, and new behaviors). In each zip file, there are two sets of .json files. The files beginning with 'calms21' contains the tracked keypoint trajectories with confidence scores and behavior annotations. The files beginning with 'taskprog_features' contains the features extracted from trajectory data from a model pre-trained on the unlabeled videos. All keypoint trajectories are produced by MARS (Segalin et al, 2020). All taskprog features are extracted from the trajectory data by TREBA (Sun et al, 2021) trained on the unlabeled videos set.

All 'calms21' files for tasks 1, 2, 3 contains the following fields:
* **`keypoints`**: tracked locations of body parts on the two interacting mice. These are produced using a Stacked Hourglass network trained on 15,000 hand-labeled frames.
  * Dimensions: (`# frames`) x (`mouse ID`) x (`x, y coordinate`) x (`body part`).
  * Units: pixels; coordinates are relative to the entire image. Original image dimensions are **1024 x 570**. 
* **`scores`**: confidence estimates for the tracked keypoints.
  * Dimensions: (`# frames`) x (`mouse ID`) x (`body part`).
  * Units: unitless, range 0 (lowest confidence) to 1 (highest confidence).  
* **`annotations`**: behaviors id as an integer annotated at each frame by a domain expert. See below in each task for the behavior id to behavior name mappings.
  * Dimensions: (`# frames`) .
* **`metadata`**: The recorded metadata is annotator_id which is represented by an int, and the vocab, containing a dictionary which maps behavior names to integer ids in `annotations`. 

Note that the 'calms21' files in the unlabeled videos set only contain the `keypoints` and `scores` field.

All 'taskprog_features' files for all tasks contains the following field:
* **`features`**: pre-computed features from a model trained with task programming on the trajectory data of the CalMS21 unlabeled videos set.
  * Dimensions: (`# frames`) x (`feature dim = 32`).

>NOTE: for all keypoints, mouse 0 is the resident (black) mouse and mouse 1 is the intruder (white) mouse. There are 7 tracked body parts, ordered (nose, left ear, right ear, neck, left hip, right hip, tail base).

We provide a conversion file `calms21_convert_to_npy.py` that converts .json files to .npy files. The .npy files can be directly used as input to our baseline code: https://gitlab.aicrowd.com/aicrowd/research/mab-e/mab-e-baselines/-/tree/master

### CalMS21 Unlabeled Videos

This set contains the unlabeled videos, and so the 'calms21' files only contain the keypoints field. The layout of the 'calms21' .json files are as follows:
```json
{
    "unlabeled_videos": {
        "unlabeled_videos/mouse142_unlabeled": {
            "keypoints": [],
            "scores": []
        },
        "unlabeled_videos/mouse143_unlabeled": {
            "keypoints": [],
            "scores": []        
        },
    }
}
```

The 'taskprog_features'.json files have the same layout, except containing `features` instead of `keypoints` and `scores`.

This set is divided into four parts for easier loading of data subsets.

### CalMS21 Task 1: Classic Classification

This set contains train and test sets for the classic classification task, where the train and test set are annotated by the same annotator for the same behaviors. The behavior mapping is: 
```
{'attack': 0, 'investigation': 1, 'mount': 2, 'other': 3}
```

The layout of the 'calms21' .json files are as follows:
```json
{
    "annotator-id_0":{
        "task1/train/mouse001_task1_annotator1": {
            "keypoints": [],
            "scores": [],
            "annotations": [],
            "metadata": {}
        },
        "task1/train/mouse002_task1_annotator1": {
            "keypoints": [],
            "scores": [],
            "annotations": [],
            "metadata": {}        
        },
    }
}
```

The 'taskprog_features'.json files have the same layout, except containing `features` instead of `keypoints`, `scores`, and `annotations`.


### CalMS21 Task 2: Annotation Style Transfer
This set contains the train and test sets for the annotation style transfer task, where there is limited training data available for the same behavior annotations from 5 different annotators. The behavior mapping is the same as Task 1. 

The layout of the 'calms21' .json files are as follows:
```json
{
    "annotator-id_1":{
        "task2/annotator1/train/mouse001_task2_annotator1": {
            "keypoints": [],
            "scores": [],
            "annotations": [],
            "metadata": {}
        },
        "task2/annotator1/train/mouse002_task2_annotator1": {
            "keypoints": [],
            "scores": [],
            "annotations": [],
            "metadata": {}      
        },
    },
    "annotator-id_2":{
        "task2/annotator2/train/mouse002_task2_annotator2": {
            "keypoints": [],
            "scores": [],
            "annotations": [],
            "metadata": {}
        },
        "task2/annotator2/train/mouse002_task2_annotator2": {
            "keypoints": [],
            "scores": [],
            "annotations": [],
            "metadata": {}        
        },
    },    
}
```

The 'taskprog_features'.json files have the same layout, except containing `features` instead of `keypoints`, `scores`, and `annotations`.


### CalMS21 Task 3: New Behaviors
Task 3 is annotated with 7 new behaviors not contained in tasks 1 and 2. Each behavior is split into train and test.

The behavior mapping is: 
```
{'other': 0, 'behavior_name': 1}
```
Here the behavior_name corresponds to the name of the first key in the dictionary, one of (approach, disengaged, groom, intromission, mount_attempt, sniff_face, whiterearing).

The layout of the 'calms21' .json files are as follows:
```json
{
    "approach":{
        "task3/approach/train/mouse001_task3_approach": {
            "keypoints": [],
            "scores": [],
            "annotations": [],
            "metadata": {}
        },
        "task3/approach/train/mouse002_task3_approach": {
            "keypoints": [],
            "scores": [],
            "annotations": [],
            "metadata": {}       
        },
    },
    "disengaged":{
        "task3/disengaged/train/mouse001_task3_disengaged": {
            "keypoints": [],
            "scores": [],
            "annotations": [],
            "metadata": {}
        },
        "task3/disengaged/train/mouse002_task3_disengaged": {
            "keypoints": [],
            "scores": [],
            "annotations": [],
            "metadata": {}        
        },
    },    
}
```

The 'taskprog_features'.json files have the same layout, except containing `features` instead of `keypoints`, `scores`, and `annotations`.


## Data collection process

### Behavior annotation

Behaviors were annotated on a frame-by-frame basis by a trained human expert in the Anderson lab. Annotators were provided with simultaneous top- and front-view video of interacting mice, and scored every video frame for close investigation, attack, and mounting (for full criteria see Methods of Segalin et al, 2020). In some videos, additional behaviors were also annotated--when this occurred, these behaviors were assigned to one of close investigation, attack, mounting, or “other” for the purpose of training classifiers. Annotation was performed either in BENTO (Segalin et al, 2020) or using a previously developed custom Matlab interface.

### Pose estimation

The poses of mice in top-view recordings are estimated using the Mouse Action Recognition System (MARS, Segalin et al, 2020), a computer vision tool that identifies seven anatomically defined keypoints on the body of each mouse: the nose, ears, base of neck, hips, and tail. For details on the pose estimation process, please refer to the MARS manuscript. (Note that while front-view video is referenced in the manuscript, pose information from the front view was not included in this dataset as it was not found to improve MARS classifier performance. This is likely due to the poor quality of front-view pose estimates due to high occurrence of occlusion as mice are interacting.)
