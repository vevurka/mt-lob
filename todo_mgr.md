# TODO

1. abstrakt po polsku i po angielsku
2. czas teraźniejszy lub przyszły
3. poprawić pseduokody by były spójne

## Introduction

9. There is no "motivation" section, should it be somewhere?

9. TODO: stress that many stock markets are operating on LOBs nowadays \cite{lob}


## LOB

1. It is an important measure of what is current state of stock, because orders which prices are the closest to \textbf{mid price} have the best chance of being executed in the next ticks. - find citation
2. related research needs some work

## Base


0. Time series definition - fix the first sentence, stationary time series instead of this stuff

1. Improve ROC area score.
TODO: draw single neuron and describe how it works.
0. tidy up svm, pca, lstm and ann

2. what is gate, RNN and vanishing gradient problem

4. something about filters?

## Problem

1. TODO: what about time series? here or in challenges
2. TODO: here why it is so hard to extract features, some cite.

## Feature extraction

1. https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf use this as motivation there 

9. might do : references to the literature -- these authors did this ref, these did this,e tc

9. TODO: this could have different order, we could start from defining mu, then define functions $f_i(x, \sigma) = ...$ and then gdfs.





## Methodology



5. LSTM architecture: TODO: make this section better
9. maybe describe conversion of the data to time steps for LSTM?

## Results

1. TODO: connect GDF feature visualizations with the rest of the text.
2. More text in introduction to results?
3. Maybe describe partial results for QUE+SVM? (there are tables already in thesis.)
4. TODO: here should be something why we do not show results without PCA and without QUE. Why do we not show results for LOG or SVM.

## Conclusions

1. feature crafting is hard - example previous queue imbalance.
2. MLP results in conclusions.
3. TODO: mention finding algorithm for finding GDF params in next steps
4. mention learning the network parameters.
5. mention using LSTM on autoencoders.

## Ideas:

1. here tell some story with the data and results. the story is that queue imbalance is a single feature and does not use time dependencies. on the other side there is no point in using more complex algorithms on this feature, because it is just not enough. But as a single feature is a very good one, so it is a good addition to the feature which grasps the shape of LOB. So we try to use more info about shape with Que, but then the complexity of the feature space is bigger. So we need more sophisticated techniques in solving this problem.

2. Logistic Regression is not suitable for more dimensions.

7. lob-ml good for SVM short description, they also use Friedman’s test.

## General with new content

1. in one place where we are overfitting show learning curve, for non overfitting and overfitting.


## Formatting

1. labels on pictures positions
2. textbf and textit for mid price, queue imbalance, previous queue imbalance.



## Done

5. choice of GDF params is in feature extraction instead of methodology
0. TODO: describe fig:me_que_algorithm_overview
1. TODO: connect mid price and indicator visualization subsections

1. TODO: check numbering if it is from 1.
2. TODO: do something about different approach to balancing classes, we want to connect it with picture showing
the ration between classes.
3. Is previous queue imbalance even described in methodology "We briefly tried to use \textbf{previous queue imbalance} feature along TODO: here we want to stress how hard is crafting features.
we briefly show stuff. svm and log are not sufficient to grasp time series dependcies?"
0. Class weights algorithm! in Classes Balancing
3. Formalize regularization
9. briefly describe architectures (many-to-one)
9.  TODO: one of gDF paramters tells us how the change of order of orders influnce the features. Talk more about these representation requirements.
1. write something about LOB requirements fullfillment for GDFs

4. GDF: TODO: here write about r and sigma 

3. gdf params choice: TODO: continue, might be that I want different plots or only one.

4. Feature choice: TODO: also tidy up the naming of GDF features, and GDF.
TODO: this plot should not be here. PCA stuff should be a bit differently dividied between here and methodology.