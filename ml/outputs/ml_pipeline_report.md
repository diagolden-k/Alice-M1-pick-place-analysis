# ML Pipeline Results: Classification / Clustering / Quality Metric

Feature set: top-25 by |Cohen's d| from phase kinematics.

## 1. Classification

Stratified 5-fold OOF comparison (primary metric: F1).

     model   oof_f1  oof_accuracy  cv_f1_gridbest                                                                                            best_params
   svm_rbf 0.943396      0.937500        0.941414                                                                 {"clf__C": 3.0, "clf__gamma": "scale"}
    logreg 0.909091      0.895833        0.908252                                                                                        {"clf__C": 0.1}
svm_linear 0.892857      0.875000        0.893100                                                                                        {"clf__C": 0.1}
        gb 0.857143      0.833333        0.853100 {"clf__learning_rate": 0.1, "clf__max_depth": 3, "clf__min_samples_leaf": 3, "clf__n_estimators": 100}
extratrees 0.852459      0.812500        0.853247                         {"clf__max_depth": null, "clf__min_samples_leaf": 1, "clf__n_estimators": 100}
        rf 0.842105      0.812500        0.844755                            {"clf__max_depth": 3, "clf__min_samples_leaf": 1, "clf__n_estimators": 100}
       knn 0.827586      0.791667        0.822244                                        {"clf__n_neighbors": 3, "clf__p": 2, "clf__weights": "uniform"}

**Selected model:** `svm_rbf` with params `{"clf__C": 3.0, "clf__gamma": "scale"}`

## 2. Clustering

Embedding: median-impute → RobustScaler → PCA(8). Selection balances silhouette and partial ARI.

            method  k  silhouette      ARI      NMI  balance  min_frac  select_score         bic
               gmm  4    0.137401 0.256952 0.330291 0.964626  0.145833      0.557992 1439.880250
            kmeans  5    0.165914 0.153337 0.262826 0.970135  0.104167      0.557209         NaN
            kmeans  3    0.198553 0.133583 0.190186 0.932048  0.229167      0.525603         NaN
agglomerative_ward  3    0.180957 0.134172 0.215471 0.817703  0.083333      0.477752         NaN
            kmeans  4    0.173538 0.136775 0.204126 0.818896  0.020833      0.255222         NaN
               gmm  5    0.159270 0.136727 0.295853 0.804316  0.041667      0.243691 1162.829182
            kmeans  6    0.162302 0.096556 0.261101 0.878636  0.020833      0.235042         NaN
agglomerative_ward  5    0.184954 0.119080 0.240192 0.754009  0.020833      0.234995         NaN
               gmm  6    0.182767 0.120516 0.266718 0.784264  0.041667      0.214815 1105.301741
agglomerative_ward  4    0.165542 0.112747 0.208443 0.705169  0.020833      0.208215         NaN
agglomerative_ward  6    0.165451 0.056357 0.239098 0.804778  0.020833      0.204579         NaN
               gmm  3    0.212715 0.210255 0.226404 0.588856  0.041667      0.173359 1390.421667

**Selected clustering:** `gmm` k=4 (silhouette=0.137, ARI=0.257)

### Cluster names

- cluster 0: **Fail_SlowLift_RoughApproach**
- cluster 1: **CleanSuccess_SmoothApproach**
- cluster 2: **Fail_RoughApproach**
- cluster 3: **CleanSuccess_FirmGrasp**

## 3. Quality Metric

Formula: `score = 100 * (0.75 * sum_i w_i * u_i + 0.25 * P_success)`

- corr(score, label) = 0.897
- mean success = 69.0, mean fail = 30.3, gap = 38.7

### Terms

- `approach_track_err_std` (invert), |d|-weight raw=1.285, q05=0.02494, q95=0.04456
- `approach_speed_maxabs` (invert), |d|-weight raw=0.824, q05=0.1138, q95=0.2143
- `approach_joint_speed_std` (invert), |d|-weight raw=1.001, q05=0.08684, q95=0.1864
- `grasp_acc_mag_mean` (as_is), |d|-weight raw=0.891, q05=0.3607, q95=1.44
- `lift_duration` (invert), |d|-weight raw=0.856, q05=0.6567, q95=3.655
- `carry_right_vy_mean` (invert), |d|-weight raw=1.037, q05=-0.09075, q95=1.136e-05
- `carry_right_vz_mean` (as_is), |d|-weight raw=0.850, q05=-0.0777, q95=-0.01661
- `retreat_vy_mean` (as_is), |d|-weight raw=0.889, q05=0.06841, q95=0.1249
- `global_jerk_mag_rms` (as_is), |d|-weight raw=0.193, q05=18.7, q95=26.71
- `place_speed_rms` (as_is), |d|-weight raw=0.439, q05=0.05805, q95=0.1239

### Normalized weights

- `approach_track_err_std`: 0.155
- `approach_speed_maxabs`: 0.100
- `approach_joint_speed_std`: 0.121
- `grasp_acc_mag_mean`: 0.108
- `lift_duration`: 0.104
- `carry_right_vy_mean`: 0.126
- `carry_right_vz_mean`: 0.103
- `retreat_vy_mean`: 0.108
- `global_jerk_mag_rms`: 0.023
- `place_speed_rms`: 0.053

## Artifacts

- `classification_comparison.csv`
- `best_classifier.joblib`
- `clustering_comparison.csv`
- `episode_results.csv` (pred, cluster, quality score)
- `quality_metric.json`
- figures: `clf_comparison.png`, `clusters_pca.png`, `quality_scores.png`