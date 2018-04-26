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
        stock = request.query_params.get('stock')
        data_length = request.query_params.get('data-length')
        data_type = request.query_params.get('data-type')
        algorithm = request.query_params.get('algorithm')

        results = Result.objects.filter(stock=stock)

        if data_length:
            results = results.filter(data_length=int(data_length))
        if data_type:
            results = results.filter(data_type=data_type)
        if algorithm == 'svm':
            gamma = request.query_params.get('gamma')
            c = request.query_params.get('c')
            coef0 = request.query_params.get('coef0')
            kernel = request.query_params.get('kernel')
            if gamma:
                results = results.filter(algorithm__svm__gamma=float(gamma))
            if c:
                results = results.filter(algorithm__svm__c=float(c))
            if coef0:
                results = results.filter(algorithm__svm__coef0=float(coef0))
            if kernel:
                results = results.filter(algorithm__svm__kernel=kernel)
        else:
            print('not found')
            return Response(status=404)
        print('returning response')
        return Response(ListSerializer(results.all(), child=ResultSerializer()).data)
