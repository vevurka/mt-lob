# Summary

Scores: MCC, ROC area. Recording F1, recall, precision, accuracy as well.
Validation method: 10-fold forward anchored cross validation.

## Queue Imbalance

* logistic regression - described in thesis
* SVM - described in thesis
* ensemble methods - briefly tried, will not show in thesis, since no good results

## Queue Imbalance + Previous Queue Imbalance

* logistic regression - no big improvement
* SVM - no big improvement

It is better for visualization how the algorithms find decision boundary. No improvement because the features
are correlated with each other.

## GDF

* logistic regression - tried very briefly, will not show in thesis, or should I?
* SVM - the results could be described in the thesis, but there is nothing interesting about them.
* MLP - the results are already described in the thesis.
* (LSTM?) - seems worth trying!

## PCA on GDF

I put all GDF features along with Queue Imbalance and Previous Queue Imbalance to PCA algorithm. Then apply 
classification.

* logistic regression - should I try? I think it is a waste of time
* SVM - tried briefly, should I show it?
* MLP - slightly better results
* (LSTM?) - from 3 stocks I tried big improvement! (even 4% of ROC area score)

### Fixes

* look at PCA explained variance, do I have to pick number of components = 10
* look at r and s parameters for GDF - design an algorithm for choosing them
* think about visualizing results (maybe there is some similar paper which has a perfect result section?)

