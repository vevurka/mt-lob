from django.conf.urls import url, include
from rest_framework import routers

from resulter import views

router = routers.DefaultRouter()
router.register(r'svm', views.SVMViewSet)
router.register(r'algorithm', views.AlgorithmViewSet)
router.register(r'result', views.ResultViewSet)


urlpatterns = [
    url(r'^', include(router.urls)),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
