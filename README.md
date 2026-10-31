# Classifying mouse social behavior from pose trajectories
An independent machine-learning analysis of the Caltech Mouse Social Interactions (CalMS21) dataset. I used tracked body landmarks from pairs of mice to classify each video frame as attack, investigation, mount, or other. I trained XGBoost classifiers and tested whether explicit measures of proximity and movement over time improved classification.

My best configuration achieved 86.98% accuracy and 79.56% macro F1 on a recording-level validation split. These are validation results, not independent test-set results or official CalMS21 benchmark scores. The central result of this project is the progression from raw pose coordinates to interaction and temporal features, and the measurable differences between those choices—not a new biological finding.
# 1. Where the data came from
Dataset: Caltech Mouse Social Interactions (CalMS21), Task 1: Classic Classification, released by the CalMS21 research team.
- Dataset record and download — CaltechDATA
- Dataset overview
- Dataset paper — The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions
- Dataset DOI: 10.22002/D1.1991
The recordings come from a resident–intruder assay: two freely behaving mice, filmed from above. The dataset authors generated pose tracks and frame-by-frame behavior annotations. I did not collect the videos, annotate the behavior, or train the original pose-tracking system. This project begins with the published keypoints and labels.

I worked with task1_classic_classification.zip, reading calms21_task1_train.json directly from the ZIP using Python's zipfile module. I did not use the raw videos, the optional MARS feature files, or the other CalMS21 tasks.

Data are not redistributed in this repository. Download the archive from the official record and keep it in data/.
# 2. Exactly what I analyzed
The training JSON contains 70 annotated recordings and 507,738 labeled frames for annotator-id_0. I split by recording, not by frame, so frames from one recording cannot appear in both training and validation.
Partition	Recordings	Frames	Role
Training	56	415,568	Fit model parameters
Validation	14	92,170	Compare features and configurations
Total analyzed	70	507,738	Published Task 1 training data


Split code: train_test_split(recording_ids, test_size=0.2, random_state=42). The split is 80/20 by recordings, not exactly by frame count.

Each frame has tracked coordinates for two mice, with seven landmarks per mouse: nose, left ear, right ear, neck, left hip, right hip, and tail base. The keypoint array for one recording has shape (frames, 2, 2, 7); the last two axes encode x/y coordinates and landmark index, respectively. Flattening those coordinates gives 28 position features per frame. Videos are sampled at 30 frames per second.

The four labels are:
Label	Behavior	All 70 recordings	Share of frames
0	Attack	14,039	2.77%
1	Investigation	146,615	28.88%
2	Mount	28,615	5.64%
3	Other	318,469	62.72%


The imbalance matters. A classifier could achieve high overall accuracy while doing poorly on Attack, which is only about 3% of the frames. For that reason, I report macro F1 and class-level precision/recall alongside accuracy.
# 3. What I did
I built the pipeline in four stages. Throughout the experiments, I kept the same recording-level train/validation split so that changes were comparable.
Stage 1 — Train the initial XGBoost model
I flattened the tracked landmarks into 28 x/y position features, then added three geometric and movement features: each mouse's neck displacement between consecutive frames (converted to pixels per second), and the neck-to-neck distance between the mice.

With these 31 features, I trained a weighted multiclass XGBoost classifier using 200 trees, maximum depth 6, learning rate 0.05, and row/column subsampling of 0.8. I computed balanced per-frame sample weights from the training labels to account for the uneven class frequencies.

This initial configuration achieved 81.32% validation accuracy and 72.08% macro F1.
Stage 2 — Represent the interaction between the mice
The original 31 features included only one explicit inter-mouse distance: neck to neck. But two mice at the same neck-to-neck distance could be facing toward or away from one another. I added three geometries that the model previously had to infer from raw coordinates:
1. Nose-to-nose distance.
2. Mouse 0's nose to mouse 1's neck.
3. Mouse 1's nose to mouse 0's neck.
That brought the feature count to 34. With the previous XGBoost settings unchanged, accuracy increased to 84.73% and macro F1 to 75.70%.

This is evidence that the additional features helped on the chosen validation recordings. It does not establish a causal biological interpretation of any single distance.
Stage 3 — Include short-term movement history
Frame-by-frame coordinates give limited information about sustained behavior. I added five temporal features, computed separately within each recording:
Added feature	Definition
Mouse 0 rolling speed	Mean neck speed over current and previous 4 frames
Mouse 1 rolling speed	Same calculation for mouse 1
Rolling mouse distance	Mean neck-to-neck distance over current and previous 4 frames
Distance change	Current minus previous neck-to-neck distance
Mouse 0 acceleration	Change in mouse 0 speed multiplied by 30 Hz


The rolling calculations use only present and past frames. For the beginning of each recording, they repeat the first value to fill the initial window rather than borrowing from a different recording. The five-frame window covers approximately 0.17 seconds of samples at 30 Hz.

With 39 total features and the same 200-tree XGBoost configuration, accuracy reached 85.77% and macro F1 reached 77.91%. Attack F1 rose from 0.45 in the 34-feature version to 0.53.
Stage 4 — Compare a small set of XGBoost configurations
Finally, I kept the 39 features fixed and compared three manually selected configurations:
Configuration	Trees	Max depth	Learning rate	Validation accuracy	Macro F1
Baseline	200	6	0.05	85.77%	77.91%
Experiment A	350	5	0.03	85.46%	77.62%
Experiment B	350	8	0.03	86.98%	79.56%


I selected Experiment B by validation macro F1, not by accuracy alone. Because the number of trees, depth, and learning rate changed together, this comparison cannot identify which parameter individually caused the improvement. It was a small manual search, not an exhaustive optimization.
# 4. What I found
Model development
Model	Features	Validation accuracy	Macro F1
XGBoost, initial	31	81.32%	72.08%
XGBoost, interaction distances	34	84.73%	75.70%
XGBoost, temporal features	39	85.77%	77.91%
XGBoost, tuned	39	86.98%	79.56%


From the first 31-feature XGBoost model to the selected 39-feature model, validation accuracy improved by 5.66 percentage points and macro F1 by 7.48 percentage points.
Best model: individual behaviors
Behavior	Precision	Recall	F1	Validation frames
Attack	0.48	0.66	0.55	3,490
Investigation	0.79	0.83	0.81	27,529
Mount	0.86	0.93	0.90	10,161
Other	0.96	0.89	0.93	50,990


Attack remains the hardest category. Its recall is about 66%, meaning approximately one-third of true attack frames are missed. Precision is 48%, meaning more than half of the frames flagged as attacks are not labeled Attack. In the normalized confusion matrix, about 29.5% of actual Attack frames are classified as Investigation. An 86.98% aggregate accuracy therefore does not mean every behavior is detected equally well.
Confusion matrix and time series
The confusion matrix is normalized by true class: each row sums to 1 and shows where actual instances of a behavior are predicted. These values are fractions of frames, not probability/confidence scores.

The following plot overlays annotated and predicted behaviors for the first 300 validation frames (10 seconds), a qualitative check for rapid label switching and mistakes around transitions.

These figures are generated by Main.py. If image links are broken, run the script or update the paths to match the files in results/.
# 5. How to run it
Requirements: Python and the Python packages below. Development used Python 3.13; other recent Python versions may also work with compatible package releases.
python -m pip install numpy scikit-learn xgboost matplotlib
1. Download task1_classic_classification.zip from CaltechDATA.
2. Place it at data/task1_classic_classification.zip. Do not unzip it; the script reads the training JSON from inside the archive.
3. Run:
python Main.py
The script loads and builds the feature matrix, trains the three candidate configurations, prints validation results, saves plots in results/, and saves the winning classifier in models/.

Important: Running Main.py retrains the models. It is not an inference-only script, and it does not evaluate the official Task 1 test set.
# 6. Files in this repository
CalMS21-Behavior-Classification/
├── README.md
├── Main.py
├── data/
│   └── task1_classic_classification.zip    # download separately; do not commit
├── models/
│   ├── mouse_behavior_xgboost_39features_baseline.json
│   └── mouse_behavior_xgboost_tuned.json
└── results/
    ├── xgboost_39features_confusion_matrix.png
    ├── xgboost_39features_predictions.png
    ├── xgboost_tuned_confusion_matrix.png
    └── xgboost_tuned_predictions.png
This tree reflects expected outputs and optional preserved baseline files; files that have not been generated or copied into the repository will not appear. The main artifact for the selected configuration is models/mouse_behavior_xgboost_tuned.json. The model file stores the trained XGBoost trees, not the Python preprocessing/feature-extraction pipeline. Making predictions on a new recording requires applying the same 39-feature extraction in the same order.

A minimal .gitignore should exclude the dataset and environment, for example:
.venv/
__pycache__/
*.pyc
data/*.zip
# 7. Tools and implementation choices
Tool	Role
Python	Complete training and evaluation workflow
NumPy	Pose arrays, distances, speeds, rolling temporal calculations
scikit-learn	Recording-level split, evaluation metrics, class weighting, confusion-matrix plotting
XGBoost	Weighted multiclass gradient-boosted tree classifier
Matplotlib	Confusion matrix and behavior-over-time figures
zipfile / json	Read the published annotation data without extracting the archive


The tuned XGBoost model uses objective="multi:softprob", tree_method="hist", n_estimators=350, max_depth=8, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, and random_state=42.
# 8. Limitations
The main limitation is model selection on one validation split. I used the same 14 recordings to compare feature sets and hyperparameters. This makes the development comparisons useful, but it can make the selected 86.98% validation accuracy optimistic as an estimate of performance on entirely new recordings. I have not performed an independent final test evaluation or recording-level cross-validation. These numbers should not be described as official benchmark results.

Frame-level metrics are not independent observations. Neighboring frames in a recording are strongly related. Although the train/validation split prevents the same recording from appearing on both sides, uncertainty cannot be assessed as if all 92,170 validation frames were independent samples. A more rigorous report would evaluate consistency across multiple held-out recordings or recording-level folds.

Pose quality was not separately audited. I relied on the published tracked coordinates and annotations. Sudden jumps or incorrect keypoints can distort velocity and acceleration. The first frame of each recording is assigned zero displacement by construction; it is not an observed stationary frame.

The features are intentionally limited. The model uses short, causal rolling windows and distances between selected landmarks. It does not learn directly from video pixels, explicitly model longer behavioral sequences, or use the provided tracking confidence scores. Additional geometry, smoothing, or sequence models might help, but those possibilities were not tested here.

The class distribution is imbalanced. Even with training sample weights, Attack has much lower precision and F1 than Mount or Other. Reporting accuracy alone would obscure this weakness.

This is not an anesthesia classifier. CalMS21 labels social behaviors in freely interacting mice. I became interested in pose-based behavior classification in connection with broader research applications, but these experiments do not measure anesthetic depth, stimulus response, or clinical state.
# References and attribution
- Sun, J. J., Karigo, T., Chakraborty, D., Mohanty, S. P., Wild, B., Sun, Q., Chen, C., Anderson, D. J., Perona, P., Yue, Y., & Kennedy, A. (2021). The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions. NeurIPS Datasets and Benchmarks. Paper.
- Caltech Mouse Social Interactions (CalMS21) Dataset, CaltechDATA. DOI: 10.22002/D1.1991.
All dataset credit belongs to the original CalMS21 authors. This repository contains my independent feature engineering, model training, evaluation, and presentation of the results; it does not claim to have produced the source mouse recordings, pose tracks, or behavior annotations.
