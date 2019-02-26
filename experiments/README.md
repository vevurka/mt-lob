# Experiments

Experiments are divided into two directories:

1. `gdf`
2. `queue_imbalance`

In each directory for each experiment `experiment_name` there is:
1. one executable python file (`[experiment_name].py`) 
2. one directory for the results (`res_[experiment_name]`)
3. one jupyter notebook visualizing the results (`[experiment_name].ipynb`)

## Queue Imbalance

Contains experiments for Queue Imbalance and Previous Queue Imbalance features:

1. que_log
2. prev_que_log
3. que_svm
4. prev_que_svm

## GDF

### Experiments

For GDF+PCA features we use LSTM, GRU, MLP and Logistic Regression. For Logistic Regression we 
calculate results in a notebook, for the rest of the algorithms the experiments are as follows:

1. Experiments which will calculate one single run of LSTM, GRU and MLP and save results
in `res_gdf_pca_lstm`, `res_gdf_pca_gru` and `res_gdf_pca_mlp` respectively:

* `$ python run_gdf_pca_lstm.py`
* `$ python run_gdf_pca_gru.py`
* `$ python run_gdf_pca_mlp.py` TODO

2. Experiments to run  LSTM, GRU or MLP classifier with the highest MCC score on 
validation set 30 times are:

* `$ python run_gdf_pca_lstm_iter.py`
* `$ python run_gdf_pca_gru_iter.py`
* `$ python run_gdf_pca_mlp_iter.py` TODO

Beware that they choose the best classifier based on results in  
`res_gdf_pca_lstm`, `res_gdf_pca_mlp` or `res_gdf_pca_gru` depending on which algorithm 
you have chosen. They save results in `res_gdf_pca_lstm_iter`, `res_gdf_pca_gru_iter` or
`res_gdf_pca_mlp_iter` respectively.

3. To run McNemar Test first you need to generate predictions for LSTM, GRU or MLP:

* `$ python run_gdf_pca_lstm_predictions.py`
* `$ python run_gdf_pca_gru_predictions.py`
* `$ python run_gdf_pca_mlp_predictions.py` TODO

They will be saved in `res_gdf_pca_lstm_mcnemar`, `res_gdf_pca_gru_mcnemar` or `res_gdf_pca_mlp_mcnemar`
respectively.

Next you can run McNemar Test, which will save results in the same directory in *.csv 
with prefix `mcnemar_':

* `$ python run_gdf_pca_lstm_mcnemar.py`
* `$ python run_gdf_pca_gru_mcnemar.py`
* `$ python run_gdf_pca_mlp_mcnemar.py` TODO

### Notebooks

#### Logistic Regression

* `gdf_pca_log.ipynb` TODO

#### LSTM 

* `gdf_pca_lstm.ipynb`
* `gdf_pca_lstm_iter.ipynb`
* `gdf_pca_lstm_mcnemar.ipynb`

#### GRU

* `gdf_pca_gru.ipynb`
* `gdf_pca_gru_iter.ipynb`
* `gdf_pca_gru_mcnemar.ipynb`

#### MLP

* `gdf_pca_mlp.ipynb`  TODO
* `gdf_pca_mlp_iter.ipynb` TODO
* `gdf_pca_mlp_mcnemar.ipynb` TODO


