# Expanded physics-pool honest selection

- pool17=17, pool40=40, pool55=61, pool70=73, full=302

## Highlight by pool size

   pool  pool_size best_honest_policy  best_honest_acc  best_honest_jaccard  topk15_acc  corr070_15_acc  fixed_acc
 pool17         17      pool17_topk15         0.875000             0.841912    0.875000        0.854167   0.875000
 pool40         40  pool40_corr085_20         0.875000             0.683411    0.791667        0.854167   0.854167
 pool55         61  pool55_corr085_20         0.812500             0.595328    0.729167        0.812500   0.895833
 pool70         73  pool70_corr070_15         0.791667             0.512429    0.708333        0.791667   0.791667
full302        302 full302_corr085_15         0.666667             0.234302         NaN             NaN        NaN

## All SVM rows (sorted)

             label  pool_size  pool_missing    policy  k model  accuracy       f1  mean_n_features  feature_jaccard
      pool55_fixed         61             0 fixed_all 61   svm  0.895833 0.905660             61.0         1.000000
      pool17_fixed         17             0 fixed_all 17   svm  0.875000 0.888889             17.0         1.000000
     pool17_topk15         17             0      topk 15   svm  0.875000 0.888889             15.0         0.841912
 pool17_family1_15         17             0   family1 15   svm  0.875000 0.888889             15.0         0.841912
 pool40_corr085_20         40             0   corr085 20   svm  0.875000 0.892857             20.0         0.683411
      pool40_fixed         40             0 fixed_all 40   svm  0.854167 0.872727             40.0         1.000000
 pool17_corr070_15         17             0   corr070 15   svm  0.854167 0.867925             13.0         0.942857
 pool17_corr085_15         17             0   corr085 15   svm  0.854167 0.867925             14.4         0.909167
 pool17_corr085_12         17             0   corr085 12   svm  0.854167 0.867925             12.0         0.759560
     pool40_topk20         40             0      topk 20   svm  0.854167 0.872727             20.0         0.741107
 pool40_corr070_15         40             0   corr070 15   svm  0.854167 0.877193             15.0         0.608927
     pool17_topk12         17             0      topk 12   svm  0.812500 0.830189             12.0         0.755604
 pool55_corr085_20         61             0   corr085 20   svm  0.812500 0.842105             20.0         0.595328
 pool55_corr070_15         61             0   corr070 15   svm  0.812500 0.847458             15.0         0.573684
      pool70_fixed         73             0 fixed_all 73   svm  0.791667 0.827586             73.0         1.000000
     pool40_topk15         40             0      topk 15   svm  0.791667 0.814815             15.0         0.575468
     pool55_topk20         61             0      topk 20   svm  0.791667 0.821429             20.0         0.567018
 pool70_corr070_15         73             0   corr070 15   svm  0.791667 0.821429             15.0         0.512429
 pool40_family1_15         40             0   family1 15   svm  0.770833 0.813559             15.0         0.598246
     pool17_topk10         17             0      topk 10   svm  0.750000 0.785714             10.0         0.704662
 pool17_corr085_10         17             0   corr085 10   svm  0.750000 0.769231             10.0         0.671329
 pool40_corr085_15         40             0   corr085 15   svm  0.750000 0.793103             15.0         0.583488
 pool70_corr085_20         73             0   corr085 20   svm  0.750000 0.785714             20.0         0.513031
     pool55_topk15         61             0      topk 15   svm  0.729167 0.779661             15.0         0.434451
     pool40_topk12         40             0      topk 12   svm  0.708333 0.758621             12.0         0.480196
     pool70_topk20         73             0      topk 20   svm  0.708333 0.740741             20.0         0.479608
 pool55_corr085_15         61             0   corr085 15   svm  0.708333 0.750000             15.0         0.447523
     pool70_topk15         73             0      topk 15   svm  0.708333 0.766667             15.0         0.388569
 pool55_corr085_10         61             0   corr085 10   svm  0.708333 0.766667             10.0         0.325672
 pool40_corr085_12         40             0   corr085 12   svm  0.687500 0.716981             12.0         0.459020
 pool55_family1_15         61             0   family1 15   svm  0.687500 0.745763             15.0         0.432570
 pool40_corr085_10         40             0   corr085 10   svm  0.687500 0.736842             10.0         0.411264
 pool70_corr085_15         73             0   corr085 15   svm  0.687500 0.736842             15.0         0.393097
 pool70_corr085_12         73             0   corr085 12   svm  0.687500 0.745763             12.0         0.343808
     pool55_topk12         61             0      topk 12   svm  0.687500 0.745763             12.0         0.340447
 pool70_corr085_10         73             0   corr085 10   svm  0.687500 0.745763             10.0         0.309005
     pool40_topk10         40             0      topk 10   svm  0.666667 0.714286             10.0         0.391090
 pool55_corr085_12         61             0   corr085 12   svm  0.666667 0.724138             12.0         0.366512
full302_corr085_15        302             0   corr085 15   svm  0.666667 0.724138             15.0         0.234302
     pool70_topk12         73             0      topk 12   svm  0.645833 0.711864             12.0         0.299215
     pool55_topk10         61             0      topk 10   svm  0.645833 0.701754             10.0         0.273786
     pool70_topk10         73             0      topk 10   svm  0.645833 0.701754             10.0         0.242204
 pool70_family1_15         73             0   family1 15   svm  0.625000 0.689655             15.0         0.361265
    full302_topk25        302             0      topk 25   svm  0.625000 0.700000             25.0         0.250091
