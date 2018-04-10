from sklearn.model_selection import GridSearchCV

degrees = np.array(range(1, 5))
cs = np.array((range(1, 5000, 100))) / 10000
gammas = list(range(1, 500, 100))

# parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':cs, 'degree': degrees, 'gamma': gammas}

parameters = {'kernel': ['rbf'], 'gamma': gammas, 'C': cs}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, scoring='roc_auc')
X = df['queue_imbalance'].values.reshape(-1, 1)
y = df['mid_price_indicator'].values
clf.fit(X, y)

from mlxtend.plotting import plot_decision_regions

for s in stocks:
    df = dfs[s]
    X = dfs[s]['queue_imbalance'].values
    Y = dfs[s]['mid_price_indicator'].values
    reg_svm = svm_classification(dfs[s], 'rbf', C=best_c[s], gamma=best_g[s])

    plot_decision_regions(X.reshape(-1, 1), Y.astype(np.integer), clf=reg_svm, legend=2, markers='x.')
    plt.figure()

from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss

cross_val_score(reg[s], dfs[s]['queue_imbalance'].values.reshape(-1, 1), 
                dfs[s]['mid_price_indicator'].values, scoring='neg_log_loss')

prob = reg[s].predict_proba(dfs[s]['queue_imbalance'].reshape(-1, 1))
log_loss(dfs[s]['mid_price_indicator'].values.reshape(-1, 1), prob[:, 1], eps=1e-15)
