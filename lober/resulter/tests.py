from django.test import TestCase

from rest_framework.test import APIClient

from resulter.models import Result, SVM, Algorithm


class TestResultSerializer(TestCase):

    def setUp(self):
        self.client = APIClient()
        self.data_svm = {'kernel': 'random', 'c': 0, 'gamma': 1, 'coef0': 2}
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
