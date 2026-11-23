# Honest selection size (k) sweep

- Rows: 1080
- Pools: {'pool17': 17, 'pool40': 40, 'pool55': 61, 'pool70': 73, 'full': 302}
- k grid: [5, 8, 10, 12, 15, 17, 20, 25, 30, 40, 50, 60]

## Sensitivity (mean over models)

  pool  policy  peak_k  peak_acc  acc_k<=12  acc_k>=30  delta_large_minus_small
  full corr070      20  0.745833   0.670833   0.683333                 0.012500
  full    topk      60  0.704167   0.641667   0.673958                 0.032292
  full family1      60  0.716667   0.612500   0.668750                 0.056250
  full corr085      40  0.741667   0.638542   0.706250                 0.067708
pool17 corr070      15  0.833333   0.755208        NaN                      NaN
pool17 corr085      15  0.829167   0.742708        NaN                      NaN
pool17 family1      15  0.850000   0.756250        NaN                      NaN
pool17    topk      15  0.850000   0.756250        NaN                      NaN
pool40 corr070      17  0.887500   0.731250   0.775000                 0.043750
pool40 corr085      17  0.862500   0.691667   0.777083                 0.085417
pool40 family1      25  0.875000   0.691667   0.810417                 0.118750
pool40    topk      25  0.900000   0.689583   0.835417                 0.145833
pool55 corr070      17  0.837500   0.703125   0.762500                 0.059375
pool55 family1      25  0.791667   0.661458   0.755208                 0.093750
pool55 corr085      20  0.820833   0.658333   0.766667                 0.108333
pool55    topk      30  0.862500   0.662500   0.822917                 0.160417
pool70 corr070      17  0.833333   0.695833   0.731250                 0.035417
pool70 family1      30  0.762500   0.666667   0.743750                 0.077083
pool70 corr085      25  0.791667   0.665625   0.750000                 0.084375
pool70    topk      30  0.804167   0.668750   0.795833                 0.127083

## Headlines by pool/band

  pool                 band  best_acc best_model best_policy  best_k  jaccard
pool17          small_k<=12  0.895833     logreg     corr070      12 0.839560
pool17          mid_k_13_25  0.937500     logreg        topk      15 0.841912
pool17  overall_best_honest  0.937500     logreg        topk      15 0.841912
pool17 fixed_all_best_model  0.937500     logreg   fixed_all      17 1.000000
pool40          small_k<=12  0.854167 svm_linear     corr070      12 0.526134
pool40          mid_k_13_25  0.937500     logreg        topk      25 0.833484
pool40          large_k>=30  0.916667     logreg        topk      30 0.858321
pool40  overall_best_honest  0.937500     logreg        topk      25 0.833484
pool40 fixed_all_best_model  0.854167    svm_rbf   fixed_all      40 1.000000
pool55          small_k<=12  0.854167     logreg     corr070      12 0.478291
pool55          mid_k_13_25  0.916667     logreg     corr070      15 0.573684
pool55          large_k>=30  0.916667     logreg        topk      30 0.825233
pool55  overall_best_honest  0.916667     logreg        topk      30 0.825233
pool55 fixed_all_best_model  0.895833    svm_rbf   fixed_all      61 1.000000
pool70          small_k<=12  0.812500     logreg     corr070      12 0.432549
pool70          mid_k_13_25  0.895833     logreg     corr070      17 0.514257
pool70          large_k>=30  0.895833     logreg        topk      30 0.687632
pool70  overall_best_honest  0.895833     logreg        topk      30 0.687632
pool70 fixed_all_best_model  0.833333        knn   fixed_all      73 1.000000
  full          small_k<=12  0.729167 svm_linear     corr070       8 0.230876
  full          mid_k_13_25  0.791667 svm_linear     corr070      20 0.334336
  full          large_k>=30  0.770833      ridge     corr085      40 0.350754
  full  overall_best_honest  0.791667 svm_linear     corr070      20 0.334336
  full fixed_all_best_model  0.770833      ridge   fixed_all     302 1.000000

## Top honest configs

  pool  policy  k_request      model  accuracy  feature_jaccard
pool17    topk         15     logreg  0.937500         0.841912
pool17 family1         17     logreg  0.937500         1.000000
pool40    topk         25     logreg  0.937500         0.833484
pool17 family1         20     logreg  0.937500         1.000000
pool17    topk         17     logreg  0.937500         1.000000
pool40 family1         25     logreg  0.937500         0.800363
pool40 corr070         15 svm_linear  0.937500         0.608927
pool17 family1         15     logreg  0.937500         0.841912
pool55    topk         40    svm_rbf  0.916667         0.799272
pool40 family1         20     logreg  0.916667         0.675072
pool55 corr070         15     logreg  0.916667         0.573684
pool40 corr070         15     logreg  0.916667         0.608927
pool40 family1         25        knn  0.916667         0.800363
pool55    topk         30     logreg  0.916667         0.825233
pool40 corr070         17        knn  0.916667         0.687102
