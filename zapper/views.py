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
from .generate_helpers import *
import json
import os
class ScanView(View):

    def get(self, request):
        return render(request, 'zapper/indexf.html')

    def post(self, request):
        """
        Reads zap codes, timestamp of sending, test code. Processes , analyzes then returns analyzed result
        """
        #reading payload as base64 strings
        payload1 = request.POST['payload1']
        #payload2 = request.POST['payload2']
        #payload3 = request.POST['payload3']
        #payload4 = request.POST['payload4']
        
        #reading test code. 4 different test codes are there designated with 1 to 4
        test_code = int(request.POST['test_code'])
        
        #capturing current time
        now = datetime.now()

        #parsing timestamp of the request transmission from client 
        sent_time = datetime.fromtimestamp(
            int(request.POST['timestamp'][:10])
            )
        
        
        print("time lag::", now - sent_time)

        #processing payload 
        #results=[process_payload(payload1),process_payload(payload2),process_payload(payload3), process_payload(payload4)]
        results=[process_payload(payload1)]
        #analyzed=analyze_results(results)
        
        
        #return HttpResponse(count_mismatch(analyzed, test_code))
        return results


class ScanViewMobile(View):

    def get(self, request):
        return render(request, 'zapper/indexm.html')

    def post(self, request):
        #os.system("rm *.png")
        payload1 = request.POST['payload1']
        #payload2 = request.POST['payload2']
        #payload3 = request.POST['payload3']
        
        test_code = int(request.POST['test_code'])
        # timestamp = datetime.fromtimestamp(
        #                 int(request.POST['timestamp'])
        #             )
        now = datetime.now()
        sent_time = datetime.fromtimestamp(
            int(request.POST['timestamp'][:10])
            )
        
        
        print("time lag::here ", now - sent_time)
        #results=[process_payload(payload1),process_payload(payload2),process_payload(payload3)]
        result = process_payload(payload1)
        print(result)
        #has_error = error_check(result)
        return HttpResponse(result)
        
        # analyzed=analyze_results(results)
        # if len(analyzed)==48:
        #     decoded = decode(analyzed)
        #     print("dd", decoded)
        #     return HttpResponse(decoded)
        # else:    
        #     return HttpResponse(analyzed)
        #return HttpResponse(count_mismatch(analyzed, test_code))

class GenerateView(View):
    def get(self, request):
        code=request.GET['code']
        n_digit = len(code)
        if n_digit==16:
            zap=generate_zap(code)
            response = HttpResponse(content_type="image/png")
            zap.save(response, "PNG")
            return response
        else:
            return HttpResponse("wrong")

class GetROIView(View):
    def post(self, request):
        payload1 = request.POST['payload1']
        roi_metrics = process_payload_get_roi(payload1)
        return HttpResponse(json.dumps(roi_metrics))