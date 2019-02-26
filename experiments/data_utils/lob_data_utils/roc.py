import matplotlib.pyplot as plt

from lob_data_utils import lob

stocks = [
    '11399', '2645', '9069', '9063', '9926', '1472', '9094', '9270', '10166', '9061', '2822',
    '2651', '2051', '1865', '1243', '4060', '1221', '2368', '12456', '12327', '2094', '9064',
    '9034', '2748', '9761', '1956', '12098', '11244', '1113', '10795', '13061', '10887', '11234',
    '9062', '1769', '7858', '4154', '4218', '13003', '9067', '10508', '2057', '12534', '1907',
    '4481', '4549', '4618', '3035', '11867', '4851', '2730', '12713', '3757', '10470', '9265',
    '4799', '11618', '1388', '9086', '9058', '11583', '2050', '2197', '9268', '12552', '9065',
    '2602', '3161', '9074', '4736', '3459', '13113', '2290', '9269', '12059', '3879', '1229',
    '4695', '5836', '10484', '2890', '1694', '1080', '3107', '11038', '12417', '9266', '4320',
    '3022', '3388', '8080', '1431', '12255', '7843', '11714', '4575', '2028', '11946', '2813',
    '11869']


i = 0
data_length = 10000
rocs_areas = {}
plt.figure()
for s in stocks:
    try:
        print('for', s)
        d, d_cv, d_test = lob.load_prepared_data(s, data_dir='../queue_imbalance/data/prepared/',
                                                 cv=True, length=data_length)

        print('performing regressions', s)
        reg = lob.logistic_regression(d, 0, len(d))

        print('performing predictions', s)
        score = lob.plot_roc(d_test, reg, stock=s)
        rocs_areas[s] = score
        print('{} (area = {})'.format(s, score))
        i += 1
        # if i % 10 == 0:
        #     plt.savefig('plots_cv_{}.png'.format(i))
        #     plt.figure()
    except Exception as e:
        print('Exception for ', s, e)

# plt.savefig('plots_{}.png'.format(i))
print(rocs_areas)
