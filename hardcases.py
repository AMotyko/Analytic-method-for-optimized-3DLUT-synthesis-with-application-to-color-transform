import numpy as np



def hard_cases_metrics(difference_array_CIEDE_L1_LMS, difference_array_CIEDE_L1_Global):
    hc_after_lms_1 = sum(x <= 1 for x in difference_array_CIEDE_L1_LMS.ravel())
    hc_after_lms_3 = sum(x <= 3 for x in difference_array_CIEDE_L1_LMS.ravel())
    hc_after_lms_5 = sum(x <= 5 for x in difference_array_CIEDE_L1_LMS.ravel())
    hc_after_lms_max = sum(x > 5 for x in difference_array_CIEDE_L1_LMS.ravel())

    hc_after_lms_max
    hc_after_lms_5 = hc_after_lms_5 - hc_after_lms_3
    hc_after_lms_3 = hc_after_lms_3 - hc_after_lms_1

    hc_after_glb_1 = sum(x <= 1 for x in difference_array_CIEDE_L1_Global.ravel())
    hc_after_glb_3 = sum(x <= 3 for x in difference_array_CIEDE_L1_Global.ravel())
    hc_after_glb_5 = sum(x <= 5 for x in difference_array_CIEDE_L1_Global.ravel())
    hc_after_glb_max = sum(x > 5 for x in difference_array_CIEDE_L1_Global.ravel())

    hc_after_glb_max
    hc_after_glb_5 = hc_after_glb_5 - hc_after_glb_3
    hc_after_glb_3 = hc_after_glb_3 - hc_after_glb_1

    lms_score = hc_after_lms_3 + hc_after_lms_5 * 4 + hc_after_lms_max * 9
    glb_score = hc_after_glb_3 + hc_after_glb_5 * 4 + hc_after_glb_max * 9

    return [hc_after_lms_1, hc_after_lms_3, hc_after_lms_5, hc_after_lms_max, lms_score, hc_after_glb_1, hc_after_glb_3, hc_after_glb_5, hc_after_glb_max, glb_score]