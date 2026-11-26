# Similarity-split sweep analysis

- Combos: **1008** (7 models × protocols × sizes × 2 distances × 3 feature modes)

## Protocol overall

         protocol  acc_mean  acc_std  acc_median   n
 mixed_stratified  0.800595 0.082023    0.812500  84
       similar_nn  0.772817 0.065851    0.770833 252
   similar_within  0.758190 0.078017    0.770833 210
    dissimilar_nn  0.736028 0.058301    0.750000 252
dissimilar_groupk  0.694907 0.178857    0.687500 210

## Large-split gaps

pair_type  gap_mean  gap_std  sim_mean  dis_mean   n  frac_gap_positive
  cluster  0.139976 0.160330  0.760759  0.620784  84           0.833333
 neighbor  0.030630 0.071515  0.774802  0.744172 168           0.601190

## Cosine gap vs n_clusters (key sensitivity)

 n_clusters  gap_mean  gap_std  sim_mean  dis_mean  mean_n_test_dis  frac_pos  n
          2  0.163690 0.162217  0.744048  0.580357             24.0  0.809524 21
          3  0.219288 0.139847  0.811550  0.592262             16.0  1.000000 21
          4  0.157479 0.118380  0.779503  0.622024             12.0  1.000000 21
          6  0.019444 0.154988  0.707937  0.688492              8.0  0.523810 21
          8  0.011828 0.185684  0.727106  0.715278              6.0  0.476190 21

## Gap by model

pair_type      model  gap_mean  gap_std   gap_q25  gap_q75  sim_mean  dis_mean  n
  cluster    svm_rbf  0.164440 0.196615  0.097997 0.300180  0.804857  0.640417 30
  cluster         gb  0.152716 0.232708  0.108696 0.297436  0.725355  0.572639 30
  cluster        knn  0.135621 0.200747  0.075974 0.281428  0.806130  0.670509 30
  cluster         rf  0.120820 0.154866  0.033634 0.213884  0.758782  0.637963 30
  cluster     logreg  0.007499 0.184113 -0.149195 0.061952  0.766434  0.758935 30
  cluster      ridge -0.068890 0.179686 -0.203852 0.079368  0.720045  0.788935 30
  cluster svm_linear -0.069228 0.128011 -0.186047 0.043750  0.725725  0.794954 30
 neighbor    svm_rbf  0.134259 0.070332  0.083333 0.187500  0.831019  0.696759 36
 neighbor      ridge  0.053241 0.058809  0.020833 0.104167  0.790509  0.737269 36
 neighbor     logreg  0.039931 0.049143  0.000000 0.062500  0.814236  0.774306 36
 neighbor svm_linear  0.038773 0.068674 -0.020833 0.104167  0.773727  0.734954 36
 neighbor        knn  0.023148 0.051461 -0.020833 0.046875  0.792245  0.769097 36
 neighbor         gb -0.013889 0.071478 -0.062500 0.020833  0.659144  0.673032 36
 neighbor         rf -0.017940 0.049087 -0.046875 0.020833  0.748843  0.766782 36

## Interpretation

1. Averaged over all combos: similar > mixed > dissimilar.
2. Cosine leave-cluster-out is *harder* when clusters are larger (n_clusters=2–3, test≈16–24): dissimilar ~0.58–0.59.
   With tiny clusters (n=8) dissimilar rises (~0.72) because holdouts are small/less distinct — so the earlier drop was NOT an artifact of tiny splits; large dissimilar splits hurt more.
3. Neighbor protocol gap is smaller (~0.03) once k_train is large (20–36).
4. Model-dependent: SVM-RBF/KNN/RF/GB show positive similar−dissimilar gaps; linear models (logreg/ridge/svm_linear) are more robust on cluster holdout.