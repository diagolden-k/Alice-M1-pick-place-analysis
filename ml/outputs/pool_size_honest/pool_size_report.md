# Honest performance by pool size

## LogReg corr0.70

 pool_size  acc_at_k15  k_used_for_k15  peak_acc  peak_k  peak_jaccard
        17    0.895833              15  0.895833      12      0.839560
        25    0.875000              15  0.895833      12      0.744176
        30    0.916667              15  0.937500      17      0.720593
        40    0.916667              15  0.916667      15      0.608927
        50    0.916667              15  0.916667      15      0.598246
        60    0.916667              15  0.916667      15      0.573684
        70    0.895833              15  0.916667      17      0.514257
       100    0.833333              15  0.875000      17      0.507538
       150    0.770833              15  0.833333      17      0.495219
       302    0.729167              15  0.770833      20      0.334336

## Best honest / fixed by pool

 pool_size               slice  accuracy      model    policy   k  jaccard
        17          best_fixed  0.937500     logreg fixed_all  17 1.000000
        25          best_fixed  0.895833     logreg fixed_all  25 1.000000
        30          best_fixed  0.895833     logreg fixed_all  30 1.000000
        40          best_fixed  0.854167    svm_rbf fixed_all  40 1.000000
        50          best_fixed  0.833333    svm_rbf fixed_all  50 1.000000
        60          best_fixed  0.875000    svm_rbf fixed_all  60 1.000000
        70          best_fixed  0.833333        knn fixed_all  70 1.000000
       100          best_fixed  0.875000    svm_rbf fixed_all 100 1.000000
       150          best_fixed  0.833333 svm_linear fixed_all 150 1.000000
       302          best_fixed  0.770833      ridge fixed_all 302 1.000000
        17     best_honest_any  0.937500     logreg      topk  15 0.841912
        25     best_honest_any  0.916667     logreg   family1  17 0.792456
        30     best_honest_any  0.937500     logreg   corr070  17 0.720593
        40     best_honest_any  0.937500     logreg      topk  24 0.834188
        50     best_honest_any  0.916667     logreg      topk  25 0.762817
        60     best_honest_any  0.916667     logreg      topk  30 0.819551
        70     best_honest_any  0.916667     logreg   corr070  17 0.514257
       100     best_honest_any  0.895833     logreg      topk  40 0.690437
       150     best_honest_any  0.875000     logreg      topk  60 0.654812
       302     best_honest_any  0.791667 svm_linear   corr070  20 0.334336
        17 logreg_corr070_peak  0.895833     logreg   corr070  12 0.839560
        25 logreg_corr070_peak  0.895833     logreg   corr070  12 0.744176
        30 logreg_corr070_peak  0.937500     logreg   corr070  17 0.720593
        40 logreg_corr070_peak  0.916667     logreg   corr070  15 0.608927
        50 logreg_corr070_peak  0.916667     logreg   corr070  15 0.598246
        60 logreg_corr070_peak  0.916667     logreg   corr070  15 0.573684
        70 logreg_corr070_peak  0.916667     logreg   corr070  17 0.514257
       100 logreg_corr070_peak  0.875000     logreg   corr070  17 0.507538
       150 logreg_corr070_peak  0.833333     logreg   corr070  17 0.495219
       302 logreg_corr070_peak  0.770833     logreg   corr070  20 0.334336
