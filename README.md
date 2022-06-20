# Massively parallel interrogation of protein fragment secretability using SECRiFY reveals features influencing secretory system transit

This is the python code for training a convolutional neural network for secretability prediction, and to make predictions for a test set. The corresponding publication can be found at
https://www.nature.com/articles/s41467-021-26720-y

Packages used during development:
```
python 3.6
tensorflow v1.10.0
numpy v1.14.5
sklearn v0.23.1
```


# Training and predicting #

To train a neural network, you can make use of either the *P. pastoris* or *S. Cerevisiae* datasets, by specifying either `pp` or `sc` in input. Additionally, you should also specify the variable length reduction strategy (one of `global_maxp`, `k_maxp`, `gru`, `zero_padding`), as well as the test set file.

```
python main.py <training_set_name> <varlen_reduction_strategy> <test_set>
```
For example:
```
python main.py pp k_maxp data/example_test_set.csv
```

This will train a network on the `pp` or `sc` dataset, and provide predictions on the supplied datasets. By default, it will do so 10 times.