# TODO

0. repeat all results for new validation method - specifically svms on queue imbalance and mlp with pca.
use approach with r,s adjusted for the procedure.

2. entropy (f.e. svm_for_lobs mentions this [3]), t-test, rejecting hyposthesis

1. It is clear that gdf parameters can be choosen depending on a stock market prices/volumes 
(most probably volumes) - differences between gdfs (regarding params r s ) - apply some clustering to visualize?

0. comparision visualization bar plot or something like that for case study

0. in results pca gdf notebooks we need to recalculate best classification to truly compare results. - nah, maybe
11. prepare notebooks for overview of how pca on gdf looks - think about visualizations of PCA, look at 
correlations between Pcaed gdfs and mid price indicator for different r and s  

## TODO overview - case study

## TODO mgr


8. Describe Recall/Precision/TPR/FPR, Matthews Correlation Coefficient
11. Describe paper: Modeling high-frequency limit order book dynamics with support vector machines. Add it to bib.

## TODO mgr cosmetic changes

9. mid price -> Mid-Price with textbf
9. queue imbalance -> Queue Imbalance with textbf
9. Cut white edges for plots for case study
11. add h! for all tables in figures 

## TODO reading

13. Read [16]
12. Read http://www.bioinf.jku.at/publications/older/2604.pdf [15]
1. Read [13]
1. Read On stock return prediction with LSTM networks [14]
0. Read [8]. Check http://di.ulb.ac.be/map/gbonte/ftp/time_ser.pdf
0. check Prediction of hidden liquidity in the limit order book of globex futures. for the gaussian model.
1. ReRead the paper about lob: "Modeling high-frequency limit order book dynamics with support 
vector machines" [3]

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

### TODO: maybe read

1. Read [9] Stock Price Correlation Coefficient Prediction with ARIMA-LSTM Hybrid Model



## Ideas:

3. queue imbalance second feature -  what about queue imbalance on more data from LOB? maybe second position?
0. what if I take different sizes of gdfs? the middle one will be the tallest, but they will be flatter and flatter

5. Check bibliography of already used papers.

## DONE

1. Read Genetic Algorithm-Optimized Long Short-Term Memory Network for Stock Market Prediction [11]
11. Read Mid-price Prediction Based on Machine Learning Methods [5]
2. Check bibliography of [5], add [5] to bibliography

0. Stock Price Prediction Using Attention-based Multi-Input LSTM [12]
3. How prediction works is well described in [5]
3. Read Benchmark Dataset for Mid-Price - [6]
3. Describe Benchmark Dataset for Mid-Price Forecasting  [6]
4. Read [7]

0. Describe plots in Case Study chapter in mgr
10. put all plots from svm part
1. rewrite results with new validation and scoring for svm in mgr
0. svm results linear copy again results from random
0. prev que imbalance in methodology
1. prev que imbalance in case study
2. prev que imbalance in results
1. rewrite results with new validation and scoring for logistic regression in mgr
0. check plots in results for logistic in mgr
0. check plots in results for svm in mgr
0. rewrite part about validation 10 anchored forward validation
0. svm - for overview analyze support vectors
1. feature search for these 3 stocks
5. copy results from random/more_queue* notebooks to overview after calculation is done
2. fix overview notebooks for svm and logistic for queue_imb (4 notebooks) for the list return from data_utils.model
1. use proper train model function from lib
0. neural nets for r 0.01
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

0. pca_gdf_que3 train svms (I've choosen small r big s because that was the best feature for these) - naaah
1. what if we take pca from smaller range of gdf - rbf - naaah
1. what if we take pca from smaller range of gdf - sigmoid - naaah
1. test logistic regression for the pca approach - it is worse than MLP
7. C parameter for logistic regression - I'm not doing that.
0. what if we take pca of gdf and queue imbalance separatelly? - nothing
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

3. Modeling high-frequency limit order book dynamics with support  vector machines [3] - described

4. Forecasting Stock Prices from the Limit Order Book using Convolution Neural Networks [4]  - described

5. Mid-price Prediction Based on Machine Learning Methods with Technical and Quantitative Indicators [5] - described

6. Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods [6] - described

8. Evaluating and Comparing Classifiers: Review, Some Recommendations and Limitations [7] - cited

10. https://arxiv.org/pdf/1808.01560.pdf [9] - nope

9. Machine Learning Strategies for Time Series Forecasting [8]

11. Genetic Algorithm-Optimized Long Short-Term Memory Network for Stock Market Prediction [11]
0. Stock Price Prediction Using Attention-based Multi-Input LSTM [12] - described a little bit

12. Stock Market Trend Prediction Using Recurrent Convolutional Neural Networks [13]

1. Read On stock return prediction with LSTM networks [14]

12. Read http://www.bioinf.jku.at/publications/older/2604.pdf [15]

13. https://www.econstor.eu/bitstream/10419/157808/1/886576210.pdf  Deep learning with long short-term memory networks for financial market
predictions [16]

14. https://arxiv.org/abs/1701.01887 Deep Learning for Time-Series Analysis [17]

15. Neural Networks for Time Series Processing (1996) http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.45.5697

16.  Sequence to Sequence Learning with Neural Networks https://arxiv.org/abs/1409.3215

17. ftp://ftp.idsia.ch/pub/juergen/icann2001predict.pdf

18. https://machinelearningmastery.com/promise-recurrent-neural-networks-time-series-forecasting/
could be good for some introduction


10. https://machinelearningmastery.com/time-series-forecasting-supervised-learning/

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
