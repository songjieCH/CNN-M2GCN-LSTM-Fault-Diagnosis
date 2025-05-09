import math

# 输入：基准数据流 t_ref, x_ref；其他数据流 t_other, x_other
# 输出：对齐后的数据流 x_aligned
def align_data(t_ref, x_ref, t_other, x_other):
    x_aligned = []
    for i in range(len(t_ref)):
        t_i = t_ref[i]  # 基准时间点
        j = find_nearest_index(t_other, t_i)
        t_j = t_other[j]
        t_j1 = t_other[j + 1]
        x_i = x_other[j] + (x_other[j + 1] - x_other[j]) * (t_i - t_j) / (t_j1 - t_j)
        x_aligned.append(x_i)
    return x_aligned

def find_nearest_index(t_other, t_i):
    for j in range(len(t_other) - 1):
        if t_other[j] <= t_i <= t_other[j + 1]:
            return j
    return len(t_other) - 2

# 输入：对齐后的数据流 x_aligned，帧长度 N，帧移 M
# 输出：分帧后的数据 frames
def frame_data(x_aligned, N, M):
    frames = []
    L = len(x_aligned)
    K = (L - N) // M + 1
    for k in range(K):
        start = k * M
        end = start + N
        frame = x_aligned[start:end]
        frames.append(frame)
    return frames

# 输入：分帧后的数据 frames，窗函数（如汉明窗）
# 输出：加窗后的数据 windowed_frames
def apply_window(frames, window_type='hamming'):
    windowed_frames = []
    N = len(frames[0])
    if window_type == 'hamming':
        window = [0.54 - 0.46 * math.cos(2 * math.pi * n / (N - 1)) for n in range(N)]
    else:
        window = [1.0] * N
    for frame in frames:
        windowed_frame = [frame[m] * window[m] for m in range(N)]
        windowed_frames.append(windowed_frame)
    return windowed_frames


# 输入：基准数据流 t_ref, x_ref；其他数据流 t_other, x_other
#       帧长度 N，帧移 M，窗函数类型 window_type
# 输出：预处理后的帧数据 windowed_frames
def preprocess_data(t_ref, x_ref, t_other, x_other, N, M, window_type='hamming'):
    x_aligned = align_data(t_ref, x_ref, t_other, x_other)
    frames = frame_data(x_aligned, N, M)
    windowed_frames = apply_window(frames, window_type)
    return windowed_frames
