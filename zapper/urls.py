from django.conf.urls import *
from .views import *
urlpatterns = [
    url(r'scan/$', ScanView.as_view(), name="zapper-scan"),
    url(r'scanm/$', ScanViewMobile.as_view(), name="zapper-scanm"),
    url(r'generate/$', GenerateView.as_view(), name="zapper-generate"),
    url(r'getroi/$', GetROIView.as_view(), name="zapper-generate"),
    url(r'digitpage/$', DigitPageView.as_view(), name="digit-page"),
    
]