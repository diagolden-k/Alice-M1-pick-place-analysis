# Similarity-based train/test splits

- within-cluster cosine=0.790, between=1.081, ratio=1.37

## Clusters

 cluster  n  n_success  n_fail  mean_within_cosine                                           episodes
       0 11          5       6            0.820756                   19,20,21,24,30,35,41,42,44,45,48
       1 16          5      11            0.758900             1,2,3,4,5,6,7,8,9,10,11,12,13,14,22,40
       2 17         16       1            0.823576 16,17,25,26,27,29,31,32,33,34,36,37,38,39,43,46,47
       3  4          1       3            0.755974                                        15,18,23,28

## Results

           protocol             feat_mode  accuracy       f1  n_scored  mean_fold_acc  std_fold_acc  feature_jaccard
   mixed_stratified       fixed_physics17  0.875000 0.888889        48       0.873333      0.047971              NaN
   similar_nn_train       fixed_physics17  0.875000 0.888889        48       0.875000      0.330719              NaN
     similar_within       fixed_physics17  0.804348 0.808511        46       0.921333      0.143180              NaN
dissimilar_nn_train       fixed_physics17  0.708333 0.720000        48       0.708333      0.454530              NaN
  dissimilar_groupk       fixed_physics17  0.645833 0.711864        48       0.656584      0.078269              NaN
   mixed_stratified honest_corr070_pool40  0.854167 0.877193        48       0.855556      0.077936         0.608927
     similar_within honest_corr070_pool40  0.826087 0.840000        46       0.899333      0.221935         0.498589
   similar_nn_train honest_corr070_pool40  0.770833 0.807018        48       0.770833      0.420297         0.391809
dissimilar_nn_train honest_corr070_pool40  0.750000 0.777778        48       0.750000      0.433013         0.390093
  dissimilar_groupk honest_corr070_pool40  0.625000 0.709677        48       0.639121      0.154697         0.435095
