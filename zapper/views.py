from django.shortcuts import render

from django.http import HttpResponse
from django.views import View


import base64
import cv2
import numpy
import time
from datetime import timedelta, datetime
from .tests import *

from .scan_helpers import *

import json
import os
class ScanView(View):

    def get(self, request):
        return render(request, 'zapper/indexf.html')

    def post(self, request):
        payload1 = request.POST['payload1']
        result = process_payload(payload1)
        print(result)
        return HttpResponse(result)
        

class DigitPageView(View):
    def get(self, request):
        return render(request, 'zapper/digit_page.html')