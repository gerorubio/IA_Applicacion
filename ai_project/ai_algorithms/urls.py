from django.urls import path
from . import views
from . import utility

urlpatterns = [
    path('', views.aprioriAlgorithm, name='ai_apriori'),
    path('correlations', views.correlationsAlgorithm, name='ai_correlations'),
    path('clustering', views.clusteringKmeans, name='ai_clustering'),
    path('measurement', views.measurementMethod, name='ai_measurement'),
    path('logisticregresión', views.regresionMethod, name='ai_logistic_regresion'),
]
