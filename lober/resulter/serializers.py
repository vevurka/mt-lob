from rest_framework import serializers

from resulter.models import SVM, Algorithm, Result


class SVMSerializer(serializers.ModelSerializer):
    class Meta:
        model = SVM
        fields = ('kernel', 'c', 'gamma', 'coef0')


class AlgorithmSerializer(serializers.ModelSerializer):
    class Meta:
        model = Algorithm
        fields = ('name', 'svm')


class ResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = Result
        fields = ('algorithm', 'roc_auc_score')
