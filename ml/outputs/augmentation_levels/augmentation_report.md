# Trajectory augmentation level comparison

## Levels

{
  "none": {
    "n_copies": 0,
    "joint_noise": 0.0,
    "time_warp": 0.0,
    "shift_frac": 0.0
  },
  "light": {
    "n_copies": 2,
    "joint_noise": 0.01,
    "time_warp": 0.05,
    "shift_frac": 0.02
  },
  "medium": {
    "n_copies": 4,
    "joint_noise": 0.02,
    "time_warp": 0.1,
    "shift_frac": 0.04
  },
  "heavy": {
    "n_copies": 8,
    "joint_noise": 0.04,
    "time_warp": 0.18,
    "shift_frac": 0.06
  }
}

## Balance

aug_level  n_rows  n_success  n_fail  success_frac
     none      48         27      21        0.5625
    light      96         54      42        0.5625
   medium     192        108      84        0.5625
    heavy     384        216     168        0.5625

## Mean by level

aug_level feat_tag  acc_mean  acc_std  f1_mean
     none  fixed17  0.854167 0.029463 0.872703
    light  fixed17  0.812500 0.029463 0.832057
   medium  fixed17  0.854167 0.000000 0.870326
    heavy  fixed17  0.791667 0.038036 0.815934
     none honest40  0.880208 0.042949 0.895224
    light honest40  0.869792 0.026215 0.883256
   medium honest40  0.869792 0.019946 0.885049
    heavy honest40  0.791667 0.024056 0.821372

## Per model

aug_level      model feat_tag  accuracy       f1  mean_train_rows
     none        knn  fixed17  0.833333 0.857143             38.4
    light        knn  fixed17  0.791667 0.821429            115.2
   medium        knn  fixed17  0.854167 0.872727            192.0
    heavy        knn  fixed17  0.833333 0.857143            345.6
     none     logreg  fixed17  0.895833 0.909091             38.4
    light     logreg  fixed17  0.854167 0.862745            115.2
   medium     logreg  fixed17  0.854167 0.867925            192.0
    heavy     logreg  fixed17  0.770833 0.792453            345.6
     none svm_linear  fixed17  0.833333 0.851852             38.4
    light svm_linear  fixed17  0.791667 0.807692            115.2
   medium svm_linear  fixed17  0.854167 0.867925            192.0
    heavy svm_linear  fixed17  0.750000 0.777778            345.6
     none    svm_rbf  fixed17  0.854167 0.872727             38.4
    light    svm_rbf  fixed17  0.812500 0.836364            115.2
   medium    svm_rbf  fixed17  0.854167 0.872727            192.0
    heavy    svm_rbf  fixed17  0.812500 0.836364            345.6
     none        knn honest40  0.833333 0.851852             38.4
    light        knn honest40  0.833333 0.846154            115.2
   medium        knn honest40  0.854167 0.877193            192.0
    heavy        knn honest40  0.770833 0.800000            345.6
     none     logreg honest40  0.916667 0.925926             38.4
    light     logreg honest40  0.895833 0.909091            115.2
   medium     logreg honest40  0.895833 0.905660            192.0
    heavy     logreg honest40  0.812500 0.842105            345.6
     none svm_linear honest40  0.916667 0.925926             38.4
    light svm_linear honest40  0.875000 0.888889            115.2
   medium svm_linear honest40  0.875000 0.884615            192.0
    heavy svm_linear honest40  0.812500 0.836364            345.6
     none    svm_rbf honest40  0.854167 0.877193             38.4
    light    svm_rbf honest40  0.875000 0.888889            115.2
   medium    svm_rbf honest40  0.854167 0.872727            192.0
    heavy    svm_rbf honest40  0.770833 0.807018            345.6

## Delta vs none

aug_level      model feat_tag  accuracy  delta_vs_none
    light     logreg  fixed17  0.854167      -0.041667
    light     logreg honest40  0.895833      -0.020833
    light    svm_rbf  fixed17  0.812500      -0.041667
    light    svm_rbf honest40  0.875000       0.020833
    light svm_linear  fixed17  0.791667      -0.041667
    light svm_linear honest40  0.875000      -0.041667
    light        knn  fixed17  0.791667      -0.041667
    light        knn honest40  0.833333       0.000000
   medium     logreg  fixed17  0.854167      -0.041667
   medium     logreg honest40  0.895833      -0.020833
   medium    svm_rbf  fixed17  0.854167       0.000000
   medium    svm_rbf honest40  0.854167       0.000000
   medium svm_linear  fixed17  0.854167       0.020833
   medium svm_linear honest40  0.875000      -0.041667
   medium        knn  fixed17  0.854167       0.020833
   medium        knn honest40  0.854167       0.020833
    heavy     logreg  fixed17  0.770833      -0.125000
    heavy     logreg honest40  0.812500      -0.104167
    heavy    svm_rbf  fixed17  0.812500      -0.041667
    heavy    svm_rbf honest40  0.770833      -0.083333
    heavy svm_linear  fixed17  0.750000      -0.083333
    heavy svm_linear honest40  0.812500      -0.104167
    heavy        knn  fixed17  0.833333       0.000000
    heavy        knn honest40  0.770833      -0.062500
