# Experiments

Experiments are divided into two directories:

1. `gdf`
2. `queue_imbalance`

In each directory for each experiment `experiment_name` there is:
1. one executable python file (`[experiment_name].py` or `run_[experiment_name].py`) 
2. one directory for the results (`res_[experiment_name]`)
3. one jupyter notebook visualizing the results (`[experiment_name].ipynb`)

To run experiments you need data in `data` directory:
* `prepared` - data extracted from raw LOB
* `data_gdf` - data after applying GDF

The data is prepared from data from raw LOB which you should put in `data/LOB` in format:
`OrderBookSnapshots_[stock]_[month][day].csv`, for instance `OrderBookSnapshots_9061_1016.csv`.

To prepare data run `$ python prepare_data.py`. For preparing GDF data two steps are required:

1. `$ python gdf_data_preparer_normalizer.py`
2. `$ python gdf_data_preparer.py` - this may take a while


## Queue Imbalance

Contains experiments for Queue Imbalance and Previous Queue Imbalance features:
    
`$ python prev_que_log.py`
`$ python prev_que_svm_lin.py`
`$ python prev_que_svm_rbf.py`
`$ python prev_que_svm_sigmoid.py`

`$ python que_log.py`
`$ python que_svm_lin.py`
`$ python que_svm_rbf.py`
`$ python que_svm_sigmoid.py`

### Notebooks

que_svm.ipynb
res_svm_queue_imbalance-prev.ipynb TODO

## GDF

### Experiments

For GDF+PCA features we use LSTM, GRU, MLP and Logistic Regression. For Logistic Regression we 
calculate results in a notebook, for the rest of the algorithms the experiments are as follows:

1. Experiments which will calculate one single run of LSTM, GRU and MLP and save results
in `res_gdf_pca_lstm`, `res_gdf_pca_gru` and `res_gdf_pca_mlp` respectively:

* `$ python run_gdf_pca_lstm.py`
* `$ python run_gdf_pca_gru.py`
* `$ python run_gdf_pca_mlp.py`

2. Experiments to run  LSTM, GRU or MLP classifier with the highest MCC score on 
validation set 30 times are:

* `$ python run_gdf_pca_lstm_iter.py`
* `$ python run_gdf_pca_gru_iter.py`
* `$ python run_gdf_pca_mlp_iter.py`

Beware that they choose the best classifier based on results in  
`res_gdf_pca_lstm`, `res_gdf_pca_mlp` or `res_gdf_pca_gru` depending on which algorithm 
you have chosen. They save results in `res_gdf_pca_lstm_iter`, `res_gdf_pca_gru_iter` or
`res_gdf_pca_mlp_iter` respectively.

3. To run McNemar Test first you need to generate predictions for LSTM, GRU or MLP:

* `$ python gdf_pca_lstm_predictions.py`
* `$ python gdf_pca_gru_predictions.py`
* `$ python gdf_pca_mlp_predictions.py`

They will be saved in `res_gdf_pca_lstm_mcnemar`, `res_gdf_pca_gru_mcnemar` or `res_gdf_pca_mlp_mcnemar`
respectively.

Next you can run McNemar Test, which will save results in the same directory in *.csv 
with prefix `mcnemar_':

* `$ python gdf_pca_lstm_mcnemar.py`
* `$ python gdf_pca_gru_mcnemar.py`
* `$ python gdf_pca_mlp_mcnemar.py`

### Notebooks

To run any notebook make sure you run experiment for QUE+LOG (script `queue_imbalance/que_log.py`),
because it is a baseline algorithm and we compare against it. Also make sure you run the corresponding 
experiment to the notebook you wish to run.

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

* `gdf_pca_mlp.ipynb`  
* `gdf_pca_mlp_iter.ipynb` 
* `gdf_pca_mlp_mcnemar.ipynb`


## Running experiments

Make sure you have Python 3.6 and install requirements from `requirements.txt`:

`$ cd data_utils; python setup.py install'
`$ pip install -r requirements.txt`

Make sure you have jupyter-notebook installed to if you wish to run the notebooks.