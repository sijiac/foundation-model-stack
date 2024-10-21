import os

USE_FP8_ATTENTION = int(os.getenv('USE_FP8_ATTENTION', '0'))
# 0 = no fp8 quantization, SDPA
# 1 = QK rowwise, P blockwise, V tensorwise (full tensor)
# 2 = QKV tensorwise
# 3 = QKV direct cast
# 4 = QK rowwise, V bf16
# 5 = QK rowwise, P blockwise, V blockwise
# 6 = OK rowwise, P blockwise, V tensorwise (per head)
USE_HDT = bool(os.getenv('USE_HDT', '0') == '1')
USE_SMOOTH_K = bool(os.getenv('USE_SMOOTH_K', '0') == '1')

print("[FP8 ATTENTION] USE_FP8_ATTENTION = ", USE_FP8_ATTENTION)
print("[FP8 ATTENTION] USE_HDT = ", USE_HDT)
print("[FP8 ATTENTION] USE_SMOOTH_K = ", USE_SMOOTH_K)

def set_fp8_attention(value):
    global USE_FP8_ATTENTION
    USE_FP8_ATTENTION = value

def get_fp8_attention():
    return USE_FP8_ATTENTION
