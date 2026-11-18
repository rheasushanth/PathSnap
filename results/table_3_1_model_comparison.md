# Table 3.1 - Overall Model Comparison

Student-test columns are the 10 held-out student-testing trips. CV columns are group-aware 5-fold cross-validation over the 80 training trips, grouped by trip.

| Model                  | Algorithm Family       | Key Configuration                                                   |   CV Macro F1-score |   Top-1 Accuracy |   Route continuity |   Student-Test Macro F1-score |   Log Loss | Inference Time   |
|:-----------------------|:-----------------------|:--------------------------------------------------------------------|--------------------:|-----------------:|-------------------:|------------------------------:|-----------:|:-----------------|
| hist_gradient_boosting | Boosting               | max_iter=400, lr=0.05                                               |              0.9034 |           0.8501 |             0.8444 |                        0.9144 |     0.1091 | 0.59 s           |
| random_forest          | Ensemble (bagging)     | 120 trees, min_samples_leaf=20, balanced_subsample                  |              0.8921 |           0.8428 |             0.835  |                        0.8991 |     0.1486 | 0.53 s           |
| decision_tree          | Tree-based (single)    | max_depth=14, min_samples_leaf=50, balanced                         |              0.87   |           0.8417 |             0.8412 |                        0.8744 |     0.2029 | 0.03 s           |
| mlp                    | Neural network         | (64,32), early_stopping, max_iter=60, scaled, fit on 100k subsample |              0.8966 |           0.8315 |             0.8386 |                        0.9045 |     0.1222 | 0.09 s           |
| logistic_regression    | Linear / probabilistic | C=1.0, class_weight=balanced, scaled                                |              0.8172 |           0.8287 |             0.8221 |                        0.8255 |     0.2389 | 0.04 s           |
| knn                    | Instance-based         | k=25, weights=distance, scaled, fit on 50k subsample                |              0.8872 |           0.8262 |             0.8462 |                        0.8967 |     0.2362 | 403.59 s         |
