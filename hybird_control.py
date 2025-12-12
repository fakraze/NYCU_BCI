"""
EOG Real-time Viewer & Car Controller
-------------------------------------
整合了 LSL 訊號即時顯示與基於 EOG 動作的車輛控制。

控制命令:
- 左眨眼 (Diff < -500, Sum < -500): 發送 '3' (左轉 1 秒)
- 右眨眼 (Diff > 500, Sum < -500): 發送 '4' (右轉 1 秒)
- 眨雙眼 (Sum > 500, Diff~0): 發送 '1' (直走 1 秒)

使用方式：
1. 確定 LSL EEG stream 正在廣播。
2. 修改 F1_IDX/F2_IDX 和 SERIAL_PORT。
3. python eog_car_control_viewer.py
4. 戴著帽子，看中間、左看、右看、眨眼，觀察圖形變化
"""

import threading
import time
import sys
import argparse
import serial # <<< 新增：引入 serial 庫

import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, filtfilt, welch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ========== SERIAL & CONTROL SETTINGS ==========
SERIAL_PORT = "COM8"  # <<< !!! 請修改為您的藍牙/Arduino 序列埠 (例如 'COM8' 或 '/dev/ttyACM0') !!! >>>
SERIAL_BAUDRATE = 9600
CONTROL_DURATION_SEC = 0.2 # 控制命令維持 1 秒

# ========== CHANNEL INDEX ==========
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

# ==================================
# === EOG 事件偵測參數與狀態 (保持不變) ===
# ==================================
THRESHOLD_DIFF_POS = 500
THRESHOLD_DIFF_NEG = -500
THRESHOLD_SUM_POS = 500
THRESHOLD_SUM_NEG = -500
DIFF_ZERO_MAX = 200 

COOLDOWN_SEC = 1.0
LAST_DETECTION_TIME = 0.0

# ========== 極端值判斷設定 (保持不變) ==========
N_POINTS_WINDOW = 100
car_controller = None # 全域 Car Controller 實例


class CarController:
    """管理序列埠通訊、發送命令和維持控制狀態的類別"""
    def __init__(self, port, baudrate):
        self.ser = None
        self.is_connected = False
        self.last_command_time = 0.0
        self.active_command = '0'  # 預設 '0' 為停止

        try:
            self.ser = serial.Serial(port, baudrate, timeout=1, write_timeout=1)
            self.is_connected = True
            print(f"[Serial] 成功連接到 {port}")
            # 啟動背景執行緒來維持命令或發送停止
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()
        except serial.SerialException as e:
            print(f"[Serial ERROR] 無法連接到 {port}: {e}")
            print("車輛控制將會被禁用。")

    def send_command_for_duration(self, command, duration=CONTROL_DURATION_SEC):
        """設定一個命令，讓背景執行緒發送並維持一段時間。"""
        if not self.is_connected:
            return
        
        current_time = time.time()
        # 如果當前有命令正在執行，則不覆蓋
        # 這裡我們允許覆蓋，確保冷卻時間內發出的新指令能立刻響應。
        self.active_command = command
        self.last_command_time = current_time + duration
        
        # 立即輸出命令，確保即時反應
        try:
            self.ser.write(self.active_command.encode())
            # 使用 sys.stdout.write 和 \r 來實現單行更新
            sys.stdout.write(f"\r[Car] 發送: {self.active_command} (持續 {duration}s) | 狀態: 進行中... {time.strftime('%H:%M:%S')}   ")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"\r[Car] 序列埠寫入錯誤: {e}                                              \n")
            sys.stdout.flush()
            self.is_connected = False
            self.active_command = '0'

    def _control_loop(self):
        """背景執行緒：負責在命令結束後發送停止命令 '0'。"""
        while self.is_connected:
            current_time = time.time()
            if self.active_command != '0' and current_time > self.last_command_time:
                # 命令時間已到，發送停止
                try:
                    self.ser.write(b'0')
                    sys.stdout.write(f"\r[Car] 發送: 0 (停止)                                                 ")
                    sys.stdout.flush()
                except Exception:
                    self.is_connected = False
                self.active_command = '0' # 重設為停止狀態
            
            time.sleep(0.05) # 50ms 檢查一次

    def close(self):
        """關閉序列埠連線"""
        if self.is_connected and self.ser:
            try:
                self.ser.write(b'0') # 確保停止
                self.ser.close()
                print("\n[Serial] 序列埠已關閉。")
            except Exception as e:
                print(f"[Serial ERROR] 關閉失敗: {e}")


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
    global LAST_DETECTION_TIME 
    global car_controller # 使用全域 CarController

    print("Looking for EEG stream (type='EEG')...")
    streams = resolve_byprop('type', 'EEG')
    if not streams:
        print("No EEG stream found. Make sure your amplifier / OpenViBE is streaming.")
        sys.exit(1)

    # ... (LSL stream 選擇邏輯，保持原樣) ...
    selected_stream = streams[0] 
    if stream_name:
        for s in streams:
            if s.name() == stream_name:
                selected_stream = s
                break
    elif len(streams) > 1:
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

    # 3. 初始化 Car Controller
    car_controller = CarController(SERIAL_PORT, SERIAL_BAUDRATE)


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
        global LAST_DETECTION_TIME # 允許修改 LAST_DETECTION_TIME

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

        # ==================================
        # === EOG 事件偵測與控制邏輯 START ===
        # ==================================
        current_time = time.time()
        
        # 檢查是否在冷卻時間內
        if current_time - LAST_DETECTION_TIME > COOLDOWN_SEC:
            
            # --- 計算最近 N_POINTS_WINDOW 的極端值 (絕對值更大的) ---
            n_points = min(len(diff), N_POINTS_WINDOW)
            
            if n_points > 0:
                window_diff = diff[-n_points:]
                window_sum = summ[-n_points:]
                
                min_diff = np.min(window_diff)
                max_diff = np.max(window_diff)
                min_sum = np.min(window_sum)
                max_sum = np.max(window_sum)
                
                # 1. 決定用於判斷的 Diff 極端值 (取絕對值更大的)
                if abs(max_diff) > abs(min_diff):
                    extreme_diff_for_check = max_diff
                else:
                    extreme_diff_for_check = min_diff
                
                # 2. 決定用於判斷的 Sum 極端值 (取絕對值更大的)
                if abs(max_sum) > abs(min_sum):
                    extreme_sum_for_check = max_sum
                else:
                    extreme_sum_for_check = min_sum
                
                # 用於眨雙眼 Diff 接近 0 的檢查
                # 這裡使用 abs(extreme_diff_for_check) 來簡化判斷，因為它是絕對值最大的那個。
                max_abs_diff = abs(extreme_diff_for_check) 
            else:
                extreme_diff_for_check = 0.0
                extreme_sum_for_check = 0.0
                max_abs_diff = 0.0

            # --- 判斷邏輯開始 ---
            detected_event = None
            command_to_send = None # 新增：控制命令
            reported_diff = 0.0
            reported_sum = 0.0

            # 判斷 左眨眼 (Diff < -500, Sum < -500) -> 左轉 '3'
            # if extreme_diff_for_check < THRESHOLD_DIFF_NEG and extreme_sum_for_check < THRESHOLD_SUM_NEG:
            if extreme_diff_for_check < THRESHOLD_DIFF_NEG and extreme_sum_for_check < THRESHOLD_SUM_NEG:
                if extreme_diff_for_check < 0:
                    detected_event = "左眨眼 (左轉)"
                    command_to_send = '3'
                    reported_diff = extreme_diff_for_check
                    reported_sum = extreme_sum_for_check
                
            # 判斷 右眨眼 (Diff > 500, Sum < -500) -> 右轉 '4'
            elif extreme_diff_for_check > THRESHOLD_DIFF_POS and extreme_sum_for_check < THRESHOLD_SUM_NEG:
                if extreme_diff_for_check > 0:
                    detected_event = "右眨眼 (右轉)"
                    command_to_send = '4'
                    reported_diff = extreme_diff_for_check
                    reported_sum = extreme_sum_for_check
                
            # 判斷 眨雙眼 (Sum > 500) -> 直走 '1'
            elif extreme_sum_for_check > 250:
                # 檢查 Max(|Diff|) 是否接近 0
                if max_abs_diff < DIFF_ZERO_MAX: 
                    detected_event = "眨雙眼 (直走)"
                    command_to_send = '1'
                    reported_diff = extreme_diff_for_check
                    reported_sum = extreme_sum_for_check

            # 如果成功偵測到事件
            if detected_event:
                print(f"\n[{time.strftime('%H:%M:%S')}] **偵測到事件: {detected_event}** "
                      f"(Extreme Diff: {reported_diff:.2f}, Extreme Sum: {reported_sum:.2f})")
                
                # 發送命令給車子，並更新冷卻時間
                if car_controller and car_controller.is_connected:
                    # 控制命令維持 CONTROL_DURATION_SEC (1秒)
                    car_controller.send_command_for_duration(command_to_send, CONTROL_DURATION_SEC)
                else:
                    print(f"[WARN] 偵測到 {detected_event} 但序列埠未連接或已斷線。")
                
                LAST_DETECTION_TIME = current_time # 更新上次偵測時間
        # ==================================
        # === EOG 事件偵測與控制邏輯 END ===
        # ==================================


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
    print("\n*** EOG 事件偵測與車輛控制已啟用 ***")
    print(f"  - 連接埠: {SERIAL_PORT}")
    print(f"  - 冷卻時間: {COOLDOWN_SEC} 秒")
    print("  - 偵測動作 -> 控制命令 (持續 1 秒):")
    print(f"    - 左眨眼 (Diff < {THRESHOLD_DIFF_NEG}): 發送 '3' (左轉)")
    print(f"    - 右眨眼 (Diff > {THRESHOLD_DIFF_POS}): 發送 '4' (右轉)")
    print(f"    - 眨雙眼 (Sum > {THRESHOLD_SUM_POS}): 發送 '1' (直走)")

    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        pass
    finally:
        running_flag[0] = False
        time.sleep(0.2)
        if car_controller:
            car_controller.close()
        print("\nViewer and Controller stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EOG Real-time Viewer & Car Controller")
    parser.add_argument(
        "--stream", "-s",
        type=str,
        default=None,
        help="指定要連接的 LSL stream 名稱 (例如: Cygnus-081020-RawEEG)"
    )
    args = parser.parse_args()
    
    try:
        main(stream_name=args.stream)
    except ValueError as e:
        print(f"\n[FATAL ERROR] {e}")