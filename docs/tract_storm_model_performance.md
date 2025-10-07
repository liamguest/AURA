# tract_storm Model Performance

| model | phase | r2 | rmse | mae | mape | explained_variance |
| --- | --- | --- | --- | --- | --- | --- |
| knn | baseline | 0.7903 | 1676.1764 | 492.8842 | 0.7878 | 0.8007 |
| knn | tuned | 0.7026 | 1996.5244 | 557.2241 | 0.8111 | 0.7178 |
| decision_tree | baseline | 0.8103 | 1594.4768 | 362.1835 | 0.3388 | 0.8105 |
| decision_tree | tuned | 0.8444 | 1443.8461 | 338.3566 | 0.3122 | 0.8444 |
| random_forest | baseline | 0.9253 | 1000.7189 | 265.3854 | 0.2743 | 0.9255 |
| random_forest | tuned | 0.9244 | 1006.5902 | 265.5672 | 0.2736 | 0.9246 |
| bagging | baseline | 0.9283 | 980.1031 | 272.9324 | 0.2996 | 0.9285 |
| bagging | tuned | 0.9269 | 990.0003 | 261.8063 | 0.2725 | 0.9271 |