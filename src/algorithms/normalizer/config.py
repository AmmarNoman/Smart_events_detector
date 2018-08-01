# Normalization parameters

import GlobalsVar

"""
Threshold for detecting when the phone was moved inside the vehicle. Shows maximal difference between vectors of acceleration when they are considered from the same interval.
"""
block_diff_thres=100.0
"""
Threshold for detecting when the phone was moved inside the vehicle. Shows maximal difference between time of neighboring vectors of acceleration when they are considered from the same interval.
"""
block_time_thres=3.0 * GlobalsVar.EXCEL_SECOND
"""
Type of the algorithm for detecting when the phone was moved inside the vehicle
"""
adjacent=False
"""
Radius in excel time, used in acceleration data smoothing
"""
sm_radius=0.5 * GlobalsVar.EXCEL_SECOND
"""
Shows how many values will be taken for smoothing very small and large ones will be thrown aside.
"""
sm_range_part=0.5
"""
Shows how many acceleration vectors will be taken for counting mean acceleration vector, very small and large ones will be thrown aside.
"""
z_range_part=0.3
"""
Sets maximal time length (in excel time), between two speed values for which speed derivative can be calculated.
"""
speed_detection_thres=3.0 * GlobalsVar.EXCEL_SECOND
"""
Output file name, if nothing passed, will be \"<input>_norm\"
"""
output=""
"""
smoothing number.
"""
smoothRadius=4