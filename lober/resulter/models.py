from django.db import models


class SVM(models.Model):
    gamma = models.FloatField()
    c = models.FloatField()
    coef0 = models.FloatField()
    kernel = models.CharField(max_length=32)


class Algorithm(models.Model):
    name = models.CharField(null=True, max_length=128)
    svm = models.ForeignKey(SVM, null=True, on_delete=models.CASCADE)


class Result(models.Model):
    data_type = models.CharField(max_length=32)
    stock = models.CharField(max_length=32)
    algorithm = models.ForeignKey(Algorithm, on_delete=models.CASCADE)
    roc_auc_score = models.FloatField()