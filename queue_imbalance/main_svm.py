from time import sleep

from lob_data_utils import lob
from calculation.lob_svm import SVMLinear, SVMRbf, SVMSigmoid

stocks = [
    '11399', '2645', '9069', '9063', '9926', '1472', '9094', '9270',
    '10166', '9061', '2822',
    '2651', '2051', '1865', '1243', '4060', '1221', '2368', '12456', '12327', '2094', '9064',
    '9034', '2748', '9761', '1956', '12098', '11244', '1113', '10795', '13061', '10887', '11234',
    '9062', '1769', '7858', '4154', '4218', '13003', '9067', '10508', '2057', '12534', '1907',
    '4481', '4549', '4618', '3035', '11867', '4851', '2730', '12713', '3757', '10470', '9265',
    '4799', '11618', '1388', '9086', '9058', '11583', '2050', '2197', '9268', '12552', '9065',
    '2602', '3161', '9074', '4736', '3459', '13113', '2290', '9269', '12059', '3879', '1229',
    '5836', '10484', '2890', '1694', '1080', '3107', '11038', '12417', '9266', '4320',
    '3022', '3388', '8080', '1431', '12255', '11714', '4575', '2028', '11946', '2813',
    '11869']

# TODO: check what is up with '4695', '7843'
# TODO: move calculation to data-utils project

gammas = [0.0005, 0.005, 5, 50, 500, 5000]
cs = [0.0005, 0.005, 5.0, 50, 500, 1000]

coef0s = [0, 0.0005, 0.005, 5, 50, 500, 5000]
data_length = 15000


def main():
    for s in stocks:
        df, df_cv, df_test = lob.load_prepared_data(s, cv=True, length=data_length)
        if df is None:
            continue
        for c in cs:
            for g in gammas:
                for coef0 in coef0s:
                    svm = SVMSigmoid(s, df, c=c, coef0=coef0, gamma=g, data_length=data_length)
                    svm.predict(df_cv, 'cv', check=False)
                sleep(1)
                svm = SVMRbf(s, df, c=c, gamma=g, data_length=data_length)
                svm.predict(df_cv, 'cv', check=False)
            sleep(1)
            svm = SVMLinear(s, df, c=c, data_length=data_length)
            svm.predict(df_cv, 'cv', check=False)


if __name__ == '__main__':
    main()
