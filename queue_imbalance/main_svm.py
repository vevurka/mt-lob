import lob
from calculation.lob_svm import SVMLinear

stocks = ['1221']
cs = [0.0001, 0.001, 1]


def main():
    df, df_cv, df_test = lob.load_prepared_data(stocks[0], cv=True, length=None)
    for c in cs:
        svm = SVMLinear(stocks[0], df, c=c)
        svm.fit()
        svm.predict(df_cv, 'cv')


# TODO: add stocks to db!!!!!!
# TODO: take care of duplicates in API
if __name__ == '__main__':
    main()