"""
EOG Real-time Viewer
--------------------

- 從 LSL 讀 EEG (type='EEG')
- 假設 Fp1 / Fp2 是你要看的水平 EOG channel
- 畫兩層：
    上：Fp1, Fp2 raw (或說已經是 EEG + EOG 混合，但重點是看相對變化)
    下：diff = Fp1 - Fp2, sum = Fp1 + Fp2

使用方式：
1. 先確定你的 LSL EEG stream 有在 broadcast (e.g., OpenViBE / LabRecorder source)
2. 修改 F1_IDX/F2_IDX 成為 Fp1/Fp2 的 channel index
3. python eog_realtime_viewer.py
4. 戴著帽子，看中間、左看、右看、眨眼，觀察圖形變化
"""

import threading
import time
import sys
import argparse

import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, filtfilt, welch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ========== CHANNEL INDEX ==========
# !!! 你要先確認這裡 index 對應到 Fp1 / Fp2 !!!
# 可以用一個小 script 印出 channel labels 再決定。
F1_IDX = 0   # Fp1
F2_IDX = 1   # Fp2

# ========== FILTER SETTINGS ==========
EOG_LOW = 0.5     # Hz
EOG_HIGH = 10.0    # Hz
FILTER_ORDER = 2

# ========== BUFFER & UPDATE ==========
BUFFER_SEC = 5.0       # 圖上顯示最近 5 秒
UPDATE_INTERVAL = 50   # 每幾 ms 更新一次圖 (50ms ≈ 20 FPS)

# ========== Y AXIS LIMITS (固定範圍，避免抖動) ==========
YLIM_RAW = (-1000, 1000)      # Fp1/Fp2 原始訊號範圍 (µV)
YLIM_FEATURE = (-1000, 1000)  # diff/sum 特徵範圍

# ========== FFT SETTINGS ==========
FFT_WINDOW_SEC = 1.0  # 1-second window for FFT
FFT_NPERSEG = 256     # Number of points per FFT segment
YLIM_FFT = (0, 100)   # FFT y 軸固定範圍 (可自行調整)

# ========== SMA & VARIANCE SETTINGS ==========
SMA_WINDOW_SEC = 1.0  # 1-second window for SMA
YLIM_VARIANCE = (0, 10000)  # Variance y 軸固定範圍 (可自行調整)

def design_eog_filter(fs, low=EOG_LOW, high=EOG_HIGH, order=FILTER_ORDER):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def lsl_reader_thread(inlet, eeg_buffer, fs, lock, running_flag):
    """持續從 LSL 讀 chunk，塞進 rolling buffer。"""
    n_channels, buffer_size = eeg_buffer.shape
    print("[Reader] started.")
    while running_flag[0]:
        try:
            chunk, ts = inlet.pull_chunk(timeout=0.2)
        except Exception as e:
            print(f"[Reader] LSL error: {e}")
            continue

        if not ts:
            continue

        data = np.asarray(chunk).T  # shape: (n_channels, n_samples_new)
        data = data[:n_channels, :]
        n_new = data.shape[1]

        with lock:
            if n_new >= buffer_size:
                eeg_buffer[:, :] = data[:, -buffer_size:]
            else:
                eeg_buffer[:, :-n_new] = eeg_buffer[:, n_new:]
                eeg_buffer[:, -n_new:] = data

        # 可以稍微 sleep 避免 100% CPU
        time.sleep(0.001)

    print("[Reader] stopped.")


def main(stream_name=None):
    print("Looking for EEG stream (type='EEG')...")
    streams = resolve_byprop('type', 'EEG')
    if not streams:
        print("No EEG stream found. Make sure your amplifier / OpenViBE is streaming.")
        sys.exit(1)

    # 列出所有找到的 streams
    print(f"\n找到 {len(streams)} 個 EEG stream(s):\n")
    for i, s in enumerate(streams):
        print(f"  [{i}] {s.name():30s} (主機: {s.hostname()})")
    
    # 選擇 stream
    selected_stream = None
    if stream_name:
        # 如果指定了名稱，尋找對應的 stream
        for s in streams:
            if s.name() == stream_name:
                selected_stream = s
                print(f"\n使用指定的 stream: {stream_name}")
                break
        if not selected_stream:
            print(f"\n警告: 找不到名稱為 '{stream_name}' 的 stream，改用第一個")
            selected_stream = streams[0]
    elif len(streams) > 1:
        # 多個 stream，讓用戶選擇
        while True:
            try:
                choice = input(f"\n請選擇要使用的 stream [0-{len(streams)-1}] (直接按 Enter 使用 [0]): ").strip()
                if choice == "":
                    selected_stream = streams[0]
                    break
                idx = int(choice)
                if 0 <= idx < len(streams):
                    selected_stream = streams[idx]
                    break
                else:
                    print(f"請輸入 0 到 {len(streams)-1} 之間的數字")
            except ValueError:
                print("請輸入有效的數字")
    else:
        # 只有一個 stream，直接使用
        selected_stream = streams[0]
    
    print(f"\n連接到: {selected_stream.name()} (主機: {selected_stream.hostname()})\n")
    inlet = StreamInlet(selected_stream)
    info = inlet.info()
    fs = info.nominal_srate()
    n_channels = info.channel_count()

    print(f"Connected to stream: {info.name()}")
    print(f"Sample rate (fs): {fs} Hz")
    print(f"Channel count   : {n_channels}")

    max_idx = max(F1_IDX, F2_IDX)
    if max_idx >= n_channels:
        raise ValueError(
            f"F1_IDX/F2_IDX exceed channel count ({n_channels}). "
            "請先確認 F1_IDX/F2_IDX 對應到 Fp1 / Fp2。"
        )

    buffer_size = int(BUFFER_SEC * fs)
    eeg_buffer = np.zeros((n_channels, buffer_size), dtype=np.float32)

    lock = threading.Lock()
    running_flag = [True]

    # 啟動 reader thread
    reader = threading.Thread(
        target=lsl_reader_thread,
        args=(inlet, eeg_buffer, fs, lock, running_flag),
        daemon=True
    )
    reader.start()

    # EOG filter
    b_eog, a_eog = design_eog_filter(fs)

    # ========== Matplotlib Figure ==========
    plt.ion()
    fig, (ax_top, ax_bottom, ax_fft, ax_variance) = plt.subplots(4, 1, figsize=(12, 10), sharex=False)

    t_axis = np.linspace(-BUFFER_SEC, 0, buffer_size)

    # 上圖：Fp1 / Fp2
    line_fp1, = ax_top.plot(t_axis, np.zeros(buffer_size), label="Fp1")
    line_fp2, = ax_top.plot(t_axis, np.zeros(buffer_size), label="Fp2", linestyle="--")
    ax_top.set_ylabel("Amplitude (µV)")
    ax_top.set_title("Fp1 / Fp2 (approx. raw EOG)")
    ax_top.legend(loc="upper right")
    ax_top.grid(True, alpha=0.3)
    ax_top.set_ylim(YLIM_RAW)  # 固定 y 軸範圍

    # 下圖：diff / sum
    line_diff, = ax_bottom.plot(t_axis, np.zeros(buffer_size), label="diff = Fp1 - Fp2")
    line_sum,  = ax_bottom.plot(t_axis, np.zeros(buffer_size), label="sum = Fp1 + Fp2", linestyle="--")
    ax_bottom.set_xlabel("Time (s, relative)")
    ax_bottom.set_ylabel("Feature")
    ax_bottom.set_title("Features (diff & sum)")
    ax_bottom.legend(loc="upper right")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.set_ylim(YLIM_FEATURE)  # 固定 y 軸範圍

    # FFT plot - 兩條線分別顯示 Fp1 和 Fp2
    line_fft1, = ax_fft.plot([], [], 'b-', label='PSD (Fp1)', linewidth=1.5)  # 藍色
    line_fft2, = ax_fft.plot([], [], 'r-', label='PSD (Fp2)', linewidth=1.5)  # 紅色
    ax_fft.set_xlabel('Frequency (Hz)')
    ax_fft.set_ylabel('Power')
    ax_fft.set_title('Power Spectral Density (1s window)')
    ax_fft.set_xlim(0, 30)  # Show 0-30Hz range
    ax_fft.set_ylim(YLIM_FFT)  # 固定 y 軸範圍
    ax_fft.legend(loc='upper right')
    ax_fft.grid(True, alpha=0.3)

    # Variance plot - 顯示 Fp1 和 Fp2 與其 SMA 的 variance
    line_var1, = ax_variance.plot([], [], 'g-', label='Var(Fp1-SMA)', linewidth=1.5)  # 綠色
    line_var2, = ax_variance.plot([], [], 'm-', label='Var(Fp2-SMA)', linewidth=1.5)  # 洋紅色
    ax_variance.set_xlabel('Time (s, relative)')
    ax_variance.set_ylabel('Variance')
    ax_variance.set_title('Variance from SMA (1s window)')
    ax_variance.set_ylim(YLIM_VARIANCE)  # 固定 y 軸範圍
    ax_variance.legend(loc='upper right')
    ax_variance.grid(True, alpha=0.3)

    plt.tight_layout()

    # ========== Animation update 函數 ==========
    def update(frame):
        with lock:
            buf = eeg_buffer.copy()

        # 取 Fp1 / Fp2
        fp1 = buf[F1_IDX, :]
        fp2 = buf[F2_IDX, :]

        # 濾波處理
        try:
            fp1_eog = filtfilt(b_eog, a_eog, fp1)
            fp2_eog = filtfilt(b_eog, a_eog, fp2)
        except Exception:
            fp1_eog = fp1
            fp2_eog = fp2

        # 更新原始訊號圖
        line_fp1.set_ydata(fp1_eog)
        line_fp2.set_ydata(fp2_eog)

        # 計算 diff/sum
        diff = fp1_eog - fp2_eog
        summ = fp1_eog + fp2_eog
        line_diff.set_ydata(diff)
        line_sum.set_ydata(summ)

        # 計算並更新 FFT
        n_fft = int(FFT_WINDOW_SEC * fs)
        if n_fft > len(fp1_eog):
            n_fft = len(fp1_eog)
        if n_fft > 0:
            freqs, psd1 = welch(fp1_eog[-n_fft:], fs=fs, nperseg=FFT_NPERSEG)
            _, psd2 = welch(fp2_eog[-n_fft:], fs=fs, nperseg=FFT_NPERSEG)
            line_fft1.set_xdata(freqs)
            line_fft1.set_ydata(psd1)
            line_fft2.set_xdata(freqs)
            line_fft2.set_ydata(psd2)

        # 計算並更新 SMA 和 Variance
        n_sma = int(SMA_WINDOW_SEC * fs)
        if n_sma > 0 and n_sma <= len(fp1_eog):
            # 計算 SMA (Simple Moving Average)
            sma_fp1 = np.convolve(fp1_eog, np.ones(n_sma)/n_sma, mode='valid')
            sma_fp2 = np.convolve(fp2_eog, np.ones(n_sma)/n_sma, mode='valid')
            
            # 對齊長度 - 取最後的部分來匹配 SMA 長度
            fp1_aligned = fp1_eog[n_sma-1:]
            fp2_aligned = fp2_eog[n_sma-1:]
            
            # 計算 variance (與 SMA 的差異的平方)
            var_fp1 = (fp1_aligned - sma_fp1) ** 2
            var_fp2 = (fp2_aligned - sma_fp2) ** 2
            
            # 創建時間軸 (對應 variance 數據長度)
            t_var = np.linspace(-len(var_fp1)/fs, 0, len(var_fp1))
            
            # 更新 variance 圖
            line_var1.set_xdata(t_var)
            line_var1.set_ydata(var_fp1)
            line_var2.set_xdata(t_var)
            line_var2.set_ydata(var_fp2)
            
            # 設定 x 軸範圍與其他圖一致
            ax_variance.set_xlim(-BUFFER_SEC, 0)

        return line_fp1, line_fp2, line_diff, line_sum, line_fft1, line_fft2, line_var1, line_var2

    ani = FuncAnimation(
        fig,
        update,
        interval=UPDATE_INTERVAL,
        blit=False
    )

    print("\nReal-time viewer running.")
    print("動作建議：")
    print("1) 先看正中 2–3 秒，觀察 baseline diff/sum")
    print("2) 然後只看左 2–3 秒，再只看右 2–3 秒，觀察 diff 正/負變化")
    print("3) 最後做幾次明顯眨眼，看 sum 的大峰值")

    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        pass
    finally:
        running_flag[0] = False
        time.sleep(0.2)
        print("Viewer stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EOG Real-time Viewer")
    parser.add_argument(
        "--stream", "-s",
        type=str,
        default=None,
        help="指定要連接的 LSL stream 名稱 (例如: Cygnus-081020-RawEEG)"
    )
    args = parser.parse_args()
    
    main(stream_name=args.stream)
