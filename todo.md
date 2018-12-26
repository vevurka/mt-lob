# TODO

0. pca_gdf_que3 train svms (I've choosen small r big s because that was the best feature for these)
1. what if we take pca from smaller range of gdf - rbf
1. what if we take pca from smaller range of gdf - sigmoid
1. test logistic regression for the pca approach

1. check what is actually picked by pca... - think about visualizations
0. differences between gdfs (regarding params r s ) - apply some clustering?
2. fix overview notebooks for svm and logistic for queue_imb (4 notebooks) for the list return from data_utils.model
5. copy results from random/more_queue* notebooks to overview after calculation is done
7. C parameter for logistic regression

## TODO overview - case study

0. svm - for overview analyze support vectors
0.check what if 15000 datapoints - for case study as well
1. feature search for these 3 stocks


## TODO mgr

0.svm results linear copy again results from random

0. prev que imbalance in methodology
1. prev que imbalance in case study
2. prev que imbalance in results
0. rewrite part about validation 10 anchored forward validation
1. rewrite results with new validation and scoring for logistic regression in mgr
0. check plots in results for logistic in mgr
0. check plots in results for svm in mgr
1. rewrite results with new validation and scoring for svm in mgr
3. Describe Benchmark Dataset for Mid-Price Forecasting  [6]
11. read Mid-price Prediction Based on Machine Learning Methods [5]
0. Describe plots in Case Study chapter in mgr
10. put all plots from svm part
  
2. Check bibilography of [5], add [5] to biblio
8. Describe Recall/Precision/TPR/FPR, Matthews Correlation Coefficient
11. Describe paper: Modeling high-frequency limit order book dynamics with support vector machines.
Add it to bib.

## TODO mgr cosmetic changes

9. mid price -> Mid-Price with textbf
9. queue imbalance -> Queue Imbalance with textbf
9. Cut white edges for plots for case study
11. add h! for all tables in figures 

## TODO reading

0. check Prediction of hidden liquidity in the limit order book of globex futures. for the gaussian model.

1. ReRead the paper about lob: "Modeling high-frequency limit order book dynamics with support 
vector machines" [3]

3. Read Benchmark Dataset for Mid-Price - [6]

3. How prediction works is well described in [5]

4. Read [7]


2. Read "Price jump prediction in Limit Order Book"

10. Read: Classification-based Financial Markets Prediction using Deep Neural Networks

12. Read:
    * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.114.4288
    * https://people.maths.ox.ac.uk/porterm/papers/gould-qf-final.pdf
    * http://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf
    * http://www.robots.ox.ac.uk/~mosb/public/pdf/115/Osborne%20et%20al.%20-%202009%20-%20Gaussian%20processes%20for%20global%20optimization.pdf

8. Learn how to describe logistic regression mathematically

9. Learn how to describe SVM mathematically

10. Learn PCA 


## Ideas:

3. queue imbalance second feature -  what about queue imbalance on more data from LOB? maybe second position?
0. what if I take different sizes of gdfs? the middle one will be the tallest, but they will be flatter and flatter

5. Check bibliography of already used papers.

## DONE

0. what about queue imbalance on gdf?

0. fill r_s_ sigmoid 9265 result

1. rerun kernels for regular gdf because I added some parameters to c and gamma
0. for small s/r notebooks pick the right features

0. sigmoid kernel for gdfs with regular scaling (for small r, small s small rs)

2. add stuff with prev_queue_imb to logistic notebook:
    * compare with regular queue_imb
    * plot it

0. regenerate notebook with svm results
0. in overview_all_svm replace results files
2. include prev queue decision region plots

0. checking if mathews score is better (for prev_queue)

3. try with 2 previous queue imbalance or queue imbalance

0. pick 3 stocks for case study (with 15000 data points): 9064 (<100), 9061 (682), 9265 (> 3000)
(mid-price means)

0. regenerate notebook with logistic regression results
4. Cohen’s kappa - metric for comparision, mean recall, precision and F1 
0. rerun feature scaling files because of the lack of the data

1. read paper "Forecasting Stock Prices from the Limit Order Book using Convolutional Neural 
Networks" [4] - useful!

3. Read Benchmark Dataset for Mid-Price Forecasting  [6]

10. Describe paper: Forecasting Stock Prices from the Limit Order Book using Convolutional Neural 
Networks [4] in the section of previous works in mgr. Add it to bib. - done

8. Describe: queueimbalanceprediction paper in mgr

2. Check how learning curve is generated - changed cv to TimeSeriesValidation

0. regenerate all results!

1. check 3 stock logistic and svm for 's'

1. min-max for lob normalization in gdf approach (the previous one is wrong because it is on past data)

1. run gdf_param_search, but fix checking if file exist

1. What about other comparision metrics - recall, precision, accuracy

2. Use cross-validation from with more folds [6]
1. Apply PCA to full GDF 

### DONE - Negative

0. what are coefs for gdf_0-50_que_prev - impossible to tell for rbf kernel 

0. use Minka’s MLE to get the number of components for PCA - nope, it wants to use all 50
0. logistic regression for pca_gdf_que3 - it doesn't seem to bring any improvement


10. trying random classifiers as well!

0. Check other features used in [3] and apply them next to queue imbalance

0. Apply filtering the noise from [4] to the datasets! Try it with GDF (SVM, logit). 
Use other metrics for comparision.
    * the noise removal is about choosing mid price indicator based on last k mid-prices
    and next k mid-prices
    * does not work for our case

0. We will look at a shape of LOB - fit linear regression to each order book. Let's start
from fitting it to prices only. We can later try to fit it to p_i, v_i or p_i * v_i

2. normalization with z-score with values from previous day for gdf (volumes and prices separately)
for gdf preparation - nope.

2. Can we use logistic regression to find gdf params and then use SVM? - nope.

3. I want to confirm that I can use Logistic Regression  to find the best params for GDF, then
tune SVM - No, we can't

4. Apply  some feature selection sklearn.feature_selection.SelectKBest to GDF - nope.


0. test approach with k-means mid-prices: 
    * what is the k?
    * is the k values same across stocks?
    * what decides about best k-value?
    * what about normalization
        - this is hard issue with that  
    -> I don't see how this approach could help with anything, it only makes it a bit smoother

## Brain dump

We start with logistic regression and queue imbalance. We use SVM with queue imbalance. 
We use SVM with queue imbalance. We check if it is better with other features, etc. We also try some 
tree algorithm on that? We do GDF. 

What is the problem we actually try to solve? How we should describe LOB for classification?
Title of thesis is "feature selection for limit order book"? 
"limit order book representation for classification"?

## Useful

3. Modeling high-frequency limit order book dynamics with support  vector machines [3]

4. Forecasting Stock Prices from the Limit Order Book using Convolutional Neural Networks [4] 

5. Mid-price Prediction Based on Machine Learning Methods with Technical and Quantitative Indicators [5]

6. Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods [6]

8. Evaluating and Comparing Classifiers: Review, Some Recommendations and Limitations [7]

1. https://stats.stackexchange.com/questions/18030/how-to-select-kernel-for-svm?rq=1

2. http://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf

3. http://www.robots.ox.ac.uk/~mosb/public/pdf/115/Osborne%20et%20al.%20-%202009%20-%20Gaussian%20processes%20for%20global%20optimization.pdf

4. This is intro:
https://www.cis.upenn.edu/~mkearns/papers/KearnsNevmyvakaHFTRiskBooks.pdf
http://epchan.blogspot.com/2013/10/how-useful-is-order-flow-and-vpin.html

5. http://jcyhong.github.io/assets/machine-learning-price-movements.pdf

6. https://stats.stackexchange.com/questions/210700/how-to-choose-between-roc-auc-and-f1-score

7. https://en.wikipedia.org/wiki/Matthews_correlation_coefficient and paper

https://github.com/rorysroes/SGX-Full-OrderBook-Tick-Data-Trading-Strategy
