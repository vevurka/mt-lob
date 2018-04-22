from rest_framework import serializers

from resulter.models import SVM, Algorithm, Result


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
        fields = ('algorithm', 'roc_auc_score')

    def create(self, validated_data):
        svm_data = validated_data.get('algorithm').get('svm')
        svm = None
        if svm_data:
            svm = SVM(kernel=svm_data.get('kernel'), c=svm_data.get('c'),
                      gamma=svm_data.get('gamma'), coef0=svm_data.get('coef0'))
            svm.save()
        algorithm_data = validated_data.get('algorithm')
        algorithm = Algorithm(name=algorithm_data.get('name'), svm=svm)
        algorithm.save()
        result = Result(roc_auc_score=validated_data.get('roc_auc_score'), algorithm=algorithm)
        result.save()
        print(validated_data)
        return result
