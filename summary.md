# Summary

Scores: MCC, ROC area. Recording F1, recall, precision, accuracy as well.
Validation method: 5-fold forward cross validation.

To compare we compare scores directly and apply McNemar test.

Methods described in thesis are **bolded**.


## Queue Imbalance

* **logistic regression** - described in thesis
* **SVM** - described in thesis
* ensemble methods - briefly tried, will not show in thesis, since no good results


## GDF

* logistic regression - tried very briefly, will not show in thesis, or should I?
* SVM - the results could be described in the thesis, but there is nothing interesting about them.
* MLP - the results could be described
* LSTM - the results are currently described in the thesis

## PCA on GDF

I put all GDF features along with Queue Imbalance and Previous Queue Imbalance to PCA algorithm. Then apply 
classification.

* logistic regression - should I try? I think it is a waste of time
* SVM - tried briefly, should I show it?
* MLP - slightly better results
* **LSTM** - described



### TODO

* what is the title of this master thesis
* could we use better architecture of LSTM?
* should we show MLP and not bolded approaches at all? or briefly explain them?
* how the results description could be improved
* how much details in description of SVM and LSTM in knowledge base?



#### Removed

## Queue Imbalance + Previous Queue Imbalance

* logistic regression - no big improvement
* SVM - no big improvement

It is better for visualization how the algorithms find decision boundary. No improvement because the features
are correlated with each other.
