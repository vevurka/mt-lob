# TODO

1. z-score for lob normalization in gdf approach (the previous one is wrong because it is on past data)
2. different stds for gaussian in gdf

3. Read Benchmark Dataset for Mid-Price Forecasting  [6]

2. Use cross-validation from with more folds [6]

1. run gdf_param_search, but fix checking if file exist


0. pick 3 stocks for case study (with 15000 data points): 9064 (<100), 9061 (682), 9265 (> 3000)
(mid-price means)

11. read Mid-price Prediction Based on Machine Learning Methods [5]


0. Describe plots in Case Study chapter in mgr

1. What about other comparision metrics - recall, precision, accuracy

0. test approach with k-means mid-prices: 
    * what is the k?
    * is the k values same across stocks?
    * what decides about best k-value?
    * what about normalization
        - this is hard issue with that  
        

10. put all plots from svm part
11. add h! for all tables in figures   

## TODO

0. check Prediction of hidden liquidity in
the limit order book of globex futures. for the gaussian model.

1. ReRead the paper about lob: "Modeling high-frequency limit order book dynamics with support 
vector machines" [3]


2. Check bibilography of [5], add [5] to biblio
3. Read Benchmark Dataset for Mid-Price - [6]

3. How prediction works is well described in [5]



2. Read "Price jump prediction in Limit Order Book"

10. Read: Classification-based Financial Markets Prediction using Deep Neural Networks

11. Describe paper: Modeling high-frequency limit order book dynamics with support vector machines.
Add it to bib.

12. Read:
    * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.114.4288
    * https://people.maths.ox.ac.uk/porterm/papers/gould-qf-final.pdf
    * http://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf
    * http://www.robots.ox.ac.uk/~mosb/public/pdf/115/Osborne%20et%20al.%20-%202009%20-%20Gaussian%20processes%20for%20global%20optimization.pdf

8. Learn how to describe logistic regression mathematically

9. Learn how to describe SVM mathematically

8. Describe Recall/Precision/TPR/FPR


## Ideas:

10. trying random classifiers as well!

1. Apply PCA to full GDF 

0. Check other features used in [3] and apply them next to queue imbalance


4. Cohen’s kappa - metric for comparision, mean recall, precision and F1 

5. Check bibliography of already used papers.

1. Does it have any sense though? GDF is a filter! Anyway we want some kind of a different filter.
Let's try rolling window instead! 



## DONE

1. read paper "Forecasting Stock Prices from the Limit Order Book using Convolutional Neural 
Networks" [4] - useful!

10. Describe paper: Forecasting Stock Prices from the Limit Order Book using Convolutional Neural 
Networks [4] in the section of previous works in mgr. Add it to bib. - done

8. Describe: queueimbalanceprediction paper in mgr

2. Check how learning curve is generated - changed cv to TimeSeriesValidation

0. regenerate all results!

1. check 3 stock logistic and svm for 's'

### DONE - Negative 

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

## Brain dump

We start with logistic regression and queue imbalance. We use SVM with queue imbalance. 
We use SVM with queue imbalance. We check if it is better with other features, etc. We also try some 
tree algorithm on that? We do GDF. We try with states and markov chains.

What is the problem we actually try to solve? How we should describe LOB for classification?
Title of thesis is "feature selection for limit order book"? 
"limit order book reperesentation for classification"?

## Useful

3. Modeling high-frequency limit order book dynamics with support  vector machines [3]

4. Forecasting Stock Prices from the Limit Order Book using Convolutional Neural Networks [4] 

5. Mid-price Prediction Based on Machine Learning Methods with Technical and Quantitative Indicators [5]

6. Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods [6]

1. https://stats.stackexchange.com/questions/18030/how-to-select-kernel-for-svm?rq=1

2. http://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf

3. http://www.robots.ox.ac.uk/~mosb/public/pdf/115/Osborne%20et%20al.%20-%202009%20-%20Gaussian%20processes%20for%20global%20optimization.pdf

4. This is intro:
https://www.cis.upenn.edu/~mkearns/papers/KearnsNevmyvakaHFTRiskBooks.pdf
http://epchan.blogspot.com/2013/10/how-useful-is-order-flow-and-vpin.html

5. http://jcyhong.github.io/assets/machine-learning-price-movements.pdf

https://github.com/rorysroes/SGX-Full-OrderBook-Tick-Data-Trading-Strategy
