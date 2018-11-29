# TODO

## States

1. wziac states.pdf
2. poklasyfikowac sobie opisane tam stany
3. zaimplementowac mdp
4. wziac queue imbalance jako stan



0. zrobic tabelke jak na str 10
1. wykres of distribution of i zbiorczy


1. tick size
0. wald test
1. likelihood test

3. check what the heck with density
4. do the same stuff for every notebook

5. sampling


1. Learn data-mining
2. Understand the problem
3. What do we want to classify?

4. Implement Queue paper.


## Questions

1. How to test this?

 logistyczna regresja ma sens (Wald test i likelihood test) 
 - chciałabym powtórzyć te testy dla naszych danych. 
 Dodatkowo po tym jak zrobiłam lepsze wykresy gęstości queue imbalance versus mid price 
 zauważyłam parę dziwnych rzeczy (np. więcej niż jedno "maksimum gęstości"),
  które chcę spróbować zbadać dokładniej biorąc pod uwagę przyszłość i przeszłość tych akcji




1. historical mid prices - clustering or classification even
3. states thingy?
0. add description of scaling to gdf in mgr


## Writting

0. What is my plan?
- what data did I use?
- what did I actually do?
- why I did that?

1. change to chapters


## GDF


What is my goal? I want to say that this approach is better than others.
Let's find the best params for the one stock.



What I want to say about this approach?
* hard to find a proper parameters and hyperparamaters
* it's prone to overfitting

## Notes

* 20 features - s=0.1, r=0.03
* 4 features - s = 0.1, r=0.02
* 10 features - s = 0.1, r=0.09 or r = 0.1

* relevant are 20 features, 50 features, 10 features middle, 4 features middle



### TODO:

1. prepare summing up notebook with different numbers of features
0. feature relevance
1. sum up the apporach of choosing r and s (looking at distributions or by just looking at not worse
stock number for each group of params)
2. present results on training dataset and testing for findings instead of random stuff
5. pick the best params and calculate on test set, compare with previous approaches
(get the logistic regression)

### In progress
1. compare different s, r params for different number of features




### Nice to have
3. compare different Ks
0. I want to have this plot with s, r and gdf


### DONE
4. calculate for 23-27, 20-30 or maybe do not
5. Let's pick two stocks first and try to find the best parameters
4. plot heatmap s vs r
0. does r have any meaning at all!? - no, and s works the best as 0.1
2. write sensible description of the approach

## SVM

kernel_choice: goal is to group stocks so they have "similar" results
kernel_choice_10000
kernel_choice_15000

compare_test_15000 - TODO: write more about plots
http://localhost:8888/notebooks/results/compare_test_15000.ipynb#RBF-TODO - finish describing kernels


1. fix size of results plots (line ones) - fixed in results_validation_best_1000


0. results per kernel on validation sets
2. reulsts per kernel on test sets

1. TODO: write equations and desriptions - in kernel_choice*
2. TODO: write conclusions in kernel_choice*

10. Check plots desriptions in kernel_choice*
11. Check all text in kernel_choice*

results_val_bests_10000

results_test_bests_15000
resutls_test_bests_5050

summary_large_mean_queue_len
summary_small_mean_queue_len


kernel_choice_5050 (optional)

## Done

1. results_val_bests_10000 - check plots  before copying


## TODO

1. write description for summary_large_mean_queue_len

2. describe parameter tunning with cv data set
3. show the results against test set
3. show also the result for logistic regression and compare it on both cv and test
5. 

mean_bid_len - check this metric!
I want to have more bests alg

1. check what about the second best kernel (especially if different kernel!)

1. Check sums of prices in queues
1. Check distribution of worse kernels
0. learning curves by kernel- pick the bestt rbf

2. https://stackoverflow.com/questions/47569394/the-graph-of-this-roc-curve-looks-strange-sklearn-svc 

# TODO: mean square error by kernel for the bests? or for all

0. learning curve, check the params for different data lengths
1. validation curve 

1. pick parameters for these 20 stocks with 5050 data separately - check the length of that data
2. apply these parameters to all other stocks

4. plot data amount diff with results fior the worst for 5050
5. put kernel to the clustering stuff 

## Done
3. compare WITH LOGISTIC RES for the same data - cv?
0. run on test!
0. write conclusions for sigmoid, poly, summary, large tick
5. Fix written stuff for summary_large_tick
6. Fix all md stuff for new notebooks
3. Learn about SVM from coursera
0. reverse large stock lists
0. Use time prediction +1
1. Clean data - use only data from 9-15 
2. Plot queue imbalance vs mid price indicator as dots
3. Fix the conclusions
4. Use SVM
0. check bests for rbf only with data length

0. rewrite conclusions for all notebooks with large tick data
1. stock characteristic describe - separate notebook
2. how to pick a kernel and params
3. heat map for rbf kernel params C and gamma
4. sigmoid kernel - how to choose parameters

0. Pick top X best stocks
1. Pick top X worst stocks
