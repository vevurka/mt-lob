from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.serializers import ListSerializer
from rest_framework.views import APIView

from resulter.models import SVM, Algorithm, Result
from resulter.serializers import SVMSerializer, AlgorithmSerializer, ResultSerializer


class SVMViewSet(viewsets.ModelViewSet):
    queryset = SVM.objects.all()  # TODO: not sorted
    serializer_class = SVMSerializer


class AlgorithmViewSet(viewsets.ModelViewSet):
    queryset = Algorithm.objects.all()  # TODO: not sorted
    serializer_class = AlgorithmSerializer


class ResultViewSet(viewsets.ModelViewSet):  # TODO: disallow get
    queryset = Result.objects.all()  # TODO: not sorted
    serializer_class = ResultSerializer
# TODO: merge 2 views


class ResultView(APIView):
    def get(self, request, format=None):
        """
        Return a list of all users.
        """
        print(request.query_params)
        stock = request.query_params.get('stock')
        data_length = request.query_params.get('data-length')
        data_type = request.query_params.get('data-type')
        algorithm = request.query_params.get('algorithm')

        results = Result.objects.filter(stock=stock)

        if data_length:
            results = Result.objects.filter(data_length=data_length)
        if data_type:
            results = Result.objects.filter(data_type=stock)
        if algorithm == 'svm':
            gamma = request.query_params.get('gamma')
            c = request.query_params.get('c')
            coef0 = request.query_params.get('coef0')
            kernel = request.query_params.get('kernel')
            if gamma:
                results = Result.objects.filter(algorithm__svm__gamma=gamma)
            if c:
                results = Result.objects.filter(algorithm__svm__c=c)
            if coef0:
                results = Result.objects.filter(algorithm__svm__coef0=coef0)
            if kernel:
                results = Result.objects.filter(algorithm__svm__kernel=kernel)
        return Response(ListSerializer(results.all(), child=ResultSerializer()).data)
