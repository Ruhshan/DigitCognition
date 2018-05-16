from django.conf.urls import *
from .views import *
urlpatterns = [
    url(r'scan/$', ScanView.as_view(), name="zapper-scan"),
    url(r'digitpage/$', DigitPageView.as_view(), name="digit-page"),
    
]