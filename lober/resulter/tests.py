from django.test import TestCase

from rest_framework.test import APIClient

from resulter.models import Result, SVM, Algorithm


class TestLoberData(TestCase):

    def setUp(self):
        self.client = APIClient()
        self.data_svm = {'kernel': 'random', 'c': 0, 'gamma': 1, 'coef0': 2}
        self.data_svm2 = {'kernel': 'random2', 'c': 1, 'gamma': 2, 'coef0': 3}
        self.data = {
            'algorithm': {
                'name': 'data-1',
                'svm': self.data_svm
            },
            'roc_auc_score': 0.5,
            'data_type': 'cv',
            'stock': '1000',
            'data_length': 0
        }
        self.data_same_stock_same_svm = {
            'algorithm': {
                'name': 'svm-random',
                'svm': self.data_svm
            },
            'roc_auc_score': 0.7,
            'data_type': 'test',
            'stock': '1000',
            'data_length': 1
        }
        self.data_same_stock = {
            'algorithm': {
                'name': 'svm-random2',
                'svm': self.data_svm2
            },
            'roc_auc_score': 0.6,
            'data_type': 'test',
            'stock': '1000',
            'data_length': 1
        }
        self.data_duplicate_svm = {
            'algorithm': {
                'name': 'data-2',
                'svm': self.data_svm
            },
            'roc_auc_score': 0.6,
            'data_type': 'cv',
            'stock': '1001',
            'data_length': 1
        }
        self.data_duplicate_alg = {
            'algorithm': {
                'name': 'data-1',
                'svm': self.data_svm
            },
            'roc_auc_score': 0.6,
            'data_type': 'cv',
            'stock': '1002',
            'data_length': 2
        }

    def tearDown(self):
        Algorithm.objects.all().delete()
        SVM.objects.all().delete()
        Result.objects.all().delete()


    def _assert_svm(self, svm: SVM, svm_dict: dict):
        self.assertEqual(svm_dict.get('gamma'), svm.gamma)
        self.assertEqual(svm_dict.get('kernel'), svm.kernel)
        self.assertEqual(svm_dict.get('c'), svm.c)
        self.assertEqual(svm_dict.get('coef0'), svm.coef0)

    def _assert_algorithm(self, alg: Algorithm, alg_dict: dict):
        self.assertEqual(alg_dict.get('name'), alg.name)
        self._assert_svm(alg.svm, alg_dict.get('svm'))

    def _assert_result(self, res: Result, res_dict: dict):
        self.assertEqual(res_dict.get('roc_auc_score'), res.roc_auc_score)
        self.assertEqual(res_dict.get('stock'), res.stock)
        self.assertEqual(res_dict.get('data_type'), res.data_type)
        self.assertEqual(res_dict.get('data_length'), res.data_length)
        self._assert_algorithm(res.algorithm, res_dict.get('algorithm'))


class TestResultSerializer(TestLoberData):

    def test_post_to_result_creates_new_svm(self):
        resp = self.client.post('/result/', data=self.data, format='json')
        result = Result.objects.get(stock='1000')

        self._assert_svm(result.algorithm.svm, self.data.get('algorithm').get('svm'))
        self.assertEqual(resp.status_code, 201)

    def test_post_to_result_creates_new_algorithm(self):
        resp = self.client.post('/result/', data=self.data, format='json')
        result = Result.objects.get(stock=self.data.get('stock'))

        self._assert_algorithm(result.algorithm, self.data.get('algorithm'))
        self.assertEqual(resp.status_code, 201)

    def test_post_to_result_creates_new_result(self):
        resp = self.client.post('/result/', data=self.data, format='json')
        result = Result.objects.get(stock=self.data.get('stock'))

        self._assert_result(result, self.data)
        self.assertEqual(resp.status_code, 201)

    def test_post_to_result_do_not_duplicate_svm(self):
        resp1 = self.client.post('/result/', data=self.data, format='json')
        result1 = Result.objects.get(stock=self.data.get('stock'))

        resp2 = self.client.post('/result/', data=self.data_duplicate_svm, format='json')
        result2 = Result.objects.get(stock=self.data_duplicate_svm.get('stock'))

        self._assert_result(result2, self.data_duplicate_svm)
        self.assertEqual(resp1.status_code, 201)
        self.assertEqual(resp2.status_code, 201)
        self.assertEqual(result1.algorithm.svm.id, result2.algorithm.svm.id)

    def test_post_to_result_do_not_duplicate_algorithm(self):
        resp1 = self.client.post('/result/', data=self.data, format='json')
        result1 = Result.objects.get(stock=self.data.get('stock'))

        resp2 = self.client.post('/result/', data=self.data_duplicate_alg, format='json')
        result2 = Result.objects.get(stock=self.data_duplicate_alg.get('stock'))

        self._assert_result(result2, self.data_duplicate_alg)
        self.assertEqual(resp1.status_code, 201)
        self.assertEqual(resp2.status_code, 201)
        self.assertEqual(result1.algorithm.id, result2.algorithm.id)

    def test_post_to_result_do_not_duplicate_result(self):
        resp1 = self.client.post('/result/', data=self.data, format='json')
        result1 = Result.objects.get(stock=self.data.get('stock'))

        resp2 = self.client.post('/result/', data=self.data, format='json')
        result2 = Result.objects.get(stock=self.data.get('stock'))

        self._assert_result(result2, self.data)
        self.assertEqual(resp1.status_code, 201)
        self.assertEqual(resp2.status_code, 201)
        self.assertEqual(result1.id, result2.id)


class TestGetResultView(TestLoberData):

    def test_get_data_for_one_stock(self):
        resp = self.client.post('/result/', data=self.data, format='json')
        get_resp = self.client.get('/get-result/?stock=' + self.data.get('stock'))

        self.assertEqual(get_resp.json(), [self.data])
        self.assertEqual(resp.status_code, 201)

    def test_get_data_for_two_same_stocks(self):
        resp_st1 = self.client.post('/result/', data=self.data, format='json')
        resp_st2 = self.client.post('/result/', data=self.data_same_stock, format='json')
        get_resp = self.client.get('/get-result/?stock=' + self.data.get('stock'))

        self.assertEqual(get_resp.json(), [self.data, self.data_same_stock])
        self.assertEqual(resp_st1.status_code, 201)
        self.assertEqual(resp_st2.status_code, 201)

    def test_get_data_for_one_stock_when_data_for_other_exist(self):
        resp_st1 = self.client.post('/result/', data=self.data, format='json')
        resp_st2 = self.client.post('/result/', data=self.data_duplicate_svm, format='json')
        get_resp = self.client.get('/get-result/?stock=' + self.data.get('stock'))

        self.assertEqual(get_resp.json(), [self.data])
        self.assertEqual(resp_st1.status_code, 201)
        self.assertEqual(resp_st2.status_code, 201)

    def test_get_data_for_not_existing_stock(self):
        resp_st1 = self.client.post('/result/', data=self.data, format='json')
        get_resp = self.client.get('/get-result/?stock=0')

        self.assertEqual(get_resp.json(), [])
        self.assertEqual(resp_st1.status_code, 201)

    def test_get_data_by_all_svm_parameters_should_get_only_exact_result(self):
        params = 'stock={}&algorithm=svm&gamma={}&c={}&coef0={}'.format(
            self.data.get('stock'),
            self.data.get('algorithm').get('svm').get('gamma'),
            self.data.get('algorithm').get('svm').get('c'),
            self.data.get('algorithm').get('svm').get('coef0'))
        resp_st1 = self.client.post('/result/', data=self.data, format='json')
        resp_st2 = self.client.post('/result/', data=self.data_same_stock, format='json')
        get_resp = self.client.get('/get-result/?' + params)

        self.assertEqual(get_resp.json(), [self.data])
        self.assertEqual(resp_st1.status_code, 201)
        self.assertEqual(resp_st2.status_code, 201)

    def test_get_data_by_gamma_svm_parameter_should_get_exact_result(self):
        params = 'stock={}&algorithm=svm&gamma={}'.format(
            self.data.get('stock'),
            self.data.get('algorithm').get('svm').get('gamma'))
        resp_st1 = self.client.post('/result/', data=self.data, format='json')
        resp_st2 = self.client.post('/result/', data=self.data_same_stock, format='json')
        get_resp = self.client.get('/get-result/?' + params)

        self.assertEqual(get_resp.json(), [self.data])
        self.assertEqual(resp_st1.status_code, 201)
        self.assertEqual(resp_st2.status_code, 201)

    def test_get_data_by_c_svm_parameter_should_get_exact_result(self):
        params = 'stock={}&algorithm=svm&c={}'.format(
            self.data.get('stock'),
            self.data.get('algorithm').get('svm').get('c'))
        resp_st1 = self.client.post('/result/', data=self.data, format='json')
        resp_st2 = self.client.post('/result/', data=self.data_same_stock, format='json')
        get_resp = self.client.get('/get-result/?' + params)

        self.assertEqual(get_resp.json(), [self.data])
        self.assertEqual(resp_st1.status_code, 201)
        self.assertEqual(resp_st2.status_code, 201)

    def test_get_data_by_coef0_svm_parameter_should_get_exact_result(self):
        params = 'stock={}&algorithm=svm&coef0={}'.format(
            self.data.get('stock'),
            self.data.get('algorithm').get('svm').get('coef0'))
        resp_st1 = self.client.post('/result/', data=self.data, format='json')
        resp_st2 = self.client.post('/result/', data=self.data_same_stock, format='json')
        get_resp = self.client.get('/get-result/?' + params)

        self.assertEqual(get_resp.json(), [self.data])
        self.assertEqual(resp_st1.status_code, 201)
        self.assertEqual(resp_st2.status_code, 201)

    def test_get_data_by_all_svm_parameters_should_get_only_exact_result_for_two_stocks(self):
        params = 'stock={}&algorithm=svm&gamma={}&c={}&coef0={}'.format(
            self.data.get('stock'),
            self.data.get('algorithm').get('svm').get('gamma'),
            self.data.get('algorithm').get('svm').get('c'),
            self.data.get('algorithm').get('svm').get('coef0'))
        resp_st1 = self.client.post('/result/', data=self.data, format='json')
        resp_st2 = self.client.post('/result/', data=self.data_same_stock_same_svm, format='json')
        get_resp = self.client.get('/get-result/?' + params)

        self.assertEqual(get_resp.json(), [self.data, self.data_same_stock_same_svm])
        self.assertEqual(resp_st1.status_code, 201)
        self.assertEqual(resp_st2.status_code, 201)