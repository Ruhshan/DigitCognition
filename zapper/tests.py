from django.test import TestCase

# Create your tests here.

def count_mismatch(detected_code, test_code):
    codes={
            1: 'stttct tccttt sctcst cstctc cstcss ttsctt',
            2: 'ccccts ctsttt ttcctc tttcts cssstc tttttc',
            3: 'cttsct stssts tstctt ssttst ssttsc stcctc',
            4: 'sscscc tststt ttsttc ttstts ttctts tsttts'
        }
    for_test = codes[test_code].replace(' ', '')
    detected = detected_code

    if len(for_test)==len(detected):
        mismatch = 0
        for i in range(len(for_test)):
            if for_test[i]!=detected[i]:
                mismatch+=1
        if mismatch>1:
            return detected + " mismatch:"+str(mismatch)
        else:
            return detected + " match"
    else:
        return detected_code

