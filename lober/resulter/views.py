from rest_framework import viewsets

from resulter.models import SVM, Algorithm, Result
from resulter.serializers import SVMSerializer, AlgorithmSerializer, ResultSerializer


class SVMViewSet(viewsets.ModelViewSet):
    queryset = SVM.objects.all()  # TODO: not sorted
    serializer_class = SVMSerializer


class AlgorithmViewSet(viewsets.ModelViewSet):
    queryset = Algorithm.objects.all()  # TODO: not sorted
    serializer_class = AlgorithmSerializer


class ResultViewSet(viewsets.ModelViewSet):
    queryset = Result.objects.all()  # TODO: not sorted
    serializer_class = ResultSerializer
