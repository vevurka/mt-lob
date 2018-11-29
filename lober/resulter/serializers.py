import logging

from rest_framework import serializers

from resulter.models import SVM, Algorithm, Result


logger = logging.getLogger(__name__)


class SVMSerializer(serializers.ModelSerializer):
    class Meta:
        model = SVM
        fields = ('kernel', 'c', 'gamma', 'coef0')


class AlgorithmSerializer(serializers.ModelSerializer):
    svm = SVMSerializer()

    class Meta:
        model = Algorithm
        fields = ('name', 'svm')


class ResultSerializer(serializers.ModelSerializer):
    algorithm = AlgorithmSerializer()

    class Meta:
        model = Result
        fields = ('algorithm', 'roc_auc_score', 'data_type', 'stock', 'data_length')

    def create(self, validated_data: dict):
        svm = self._svm(validated_data)
        algorithm = self._algorithm(validated_data, svm)
        result = self._result(validated_data, algorithm)
        logger.info('Saved result data %s', result)
        return result

    @staticmethod
    def _svm(validated_data: dict) -> SVM:
        svm_data = validated_data.get('algorithm').get('svm')
        svm = None
        if svm_data:
            svm, created = SVM.objects.get_or_create(kernel=svm_data.get('kernel'), c=svm_data.get('c'),
                                                     gamma=svm_data.get('gamma'), coef0=svm_data.get('coef0'))
            logger.info('SVM %s was created %s', svm, created)
            svm.save()
        return svm

    @staticmethod
    def _algorithm(validated_data: dict, svm: SVM) -> Algorithm:
        algorithm_data = validated_data.get('algorithm')
        algorithm, created = Algorithm.objects.get_or_create(name=algorithm_data.get('name'), svm=svm)

        logger.info('Algorithm %s was created %s', algorithm, created)
        algorithm.save()
        return algorithm

    @staticmethod
    def _result(validated_data: dict, algorithm: Algorithm) -> Result:
        result, created = Result.objects.get_or_create(
            roc_auc_score=validated_data.get('roc_auc_score'), algorithm=algorithm,
            data_type=validated_data.get('data_type'), stock=validated_data.get('stock'),
            data_length=validated_data.get('data_length'))
        logger.info('Result %s was created %s', result, created)
        result.save()
        return result
