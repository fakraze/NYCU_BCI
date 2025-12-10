from pylsl import StreamInlet, resolve_streams, resolve_byprop
import numpy as np
import time
import threading
from scipy.signal import butter, filtfilt
import serial

# ======== 設定參數 ========
SAMPLE_RATE = 1000  # 請確認這與您的 LSL stream 採樣率一致
# 根據經驗設定
thres = 20.0
CHANNEL_COUNT = 2
BUFFER_SIZE = 1000  # lpha 波需要約 1秒資料比較準 (1000 samples)
eeg_buffer = np.zeros((CHANNEL_COUNT, BUFFER_SIZE))
# 窗函數 (Hanning Window)，用於減少頻譜洩漏
# 形狀需為 (1, BUFFER_SIZE) 以便與 eeg_buffer (2, BUFFER_SIZE) 相乘
window = np.hanning(BUFFER_SIZE).reshape(1, -1)

# ser = serial.Serial("COM5", 9600, timeout=10, write_timeout=10)

# ======== EEG Reading Thread ============
def read_eeg(inlet):
    global eeg_buffer
    while True:
        sample, timestamp = inlet.pull_sample()
        if sample:
            # 取出特定 channel
            # selected_data = sample[0:2] + sample[4:6]
            selected_data = sample[4:6]
            sample_np = np.array(selected_data).reshape(-1, 1)

            # Update buffer (Rolling buffer)
            eeg_buffer[:, :-1] = eeg_buffer[:, 1:]
            eeg_buffer[:, -1] = sample_np.flatten()


# ======== 輔助函式：計算特定頻帶能量 ========
def get_band_power(psd, freq_axis, low, high):
    """
    psd: Power Spectral Density (已經平均過頻道的)
    freq_axis: 頻率軸
    low, high: 頻帶範圍
    """
    # 找出在 low ~ high 範圍內的頻率 index
    idx = np.logical_and(freq_axis >= low, freq_axis <= high)
    # 將該範圍內的能量加總 (也可以用 mean，看你習慣)
    return np.sum(psd[idx])


def main():
    # 設定濾波器：Alpha 波範圍 8-13 Hz
    lowcut = 8.0
    highcut = 13.0
    nyq = 0.5 * SAMPLE_RATE
    b, a = butter(2, [lowcut / nyq, highcut / nyq], btype='band')

    print(f"Start detecting Alpha Wave ({lowcut}-{highcut} Hz)...")
    print("Please CLOSE YOUR EYES to boost Alpha waves.")

    while True:
        # 簡單檢查 buffer 是否填滿 (避免初期雜訊)
        if np.abs(eeg_buffer[0, 0]) < 1e-6:
            # 這裡假設如果第一個值還是 0 代表還沒填滿或沒訊號
            time.sleep(0.1)
            continue

        # 1. 濾波：取出 Alpha 頻段
        # axis=1 代表對時間軸濾波
        alpha_wave = filtfilt(b, a, eeg_buffer, axis=1)

        # 2. 計算能量 (Alpha Power)
        # 方法：訊號平方 -> 取平均 (Mean Squared Power)
        # 我們計算所有 Channel 的平均能量
        # alpha_wave ** 2 把正負號都變正值
        power_per_channel = np.mean(alpha_wave ** 2, axis=1)
        avg_alpha_power = np.mean(power_per_channel)
        # ====== 1. 預處理：去直流 (Demean) 與 加窗 (Windowing) ======
        # 去除 DC offset (平均值)，避免 0Hz 能量過大
        data_detrend = eeg_buffer - np.mean(eeg_buffer, axis=1, keepdims=True)
        # 乘上窗函數
        data_windowed = data_detrend * window

        # ====== 2. FFT 運算 ======
        # rfft: Real FFT (只計算正頻率部分)
        fft_vals = np.fft.rfft(data_windowed, axis=1)

        # 計算功率譜 (PSD)
        # 取絕對值(振幅) -> 平方 -> 除以長度(正規化)
        # 這裡簡單用 |FFT|^2 / N 即可代表相對能量強度
        psd = (np.abs(fft_vals) ** 2) / BUFFER_SIZE

        # 將兩個頻道的能量平均 (O1 和 O2 平均)
        avg_psd = np.mean(psd, axis=0)

        # ====== 3. 提取頻帶能量 ======
        alpha_power = get_band_power(avg_psd, SAMPLE_RATE, 8, 13)
        beta_power = get_band_power(avg_psd, SAMPLE_RATE, 13, 30)

        ratio = alpha_power / (beta_power + 1e-6)

        # 3. 判斷觸發
        if ratio > thres:
            print(f"Move Forward! >>> Power: {avg_alpha_power:.2f}")
            # ser.write(b'1')
        else:
            print(f"Stop.           ... Power: {avg_alpha_power:.2f}")
            # ser.write(b'0')

        time.sleep(0.2)  # 降低更新頻率，方便閱讀數據


# ======== LSL Inlet Setup ============
def setup_lsl_inlet(stream_name="Cygnus-085124-RawEEG"):
    print("Resolving streams...")
    # 這裡改成先列出所有 streams 讓你知道有沒有抓到
    streams = resolve_streams()
    if not streams:
        print("No streams found!")
        exit(1)

    for i, s in enumerate(streams):
        print(f"[{i}] {s.name()} - type: {s.type()}")

    # 嘗試自動抓取包含 'EEG' 的 stream，或是指定名稱
    target_stream = streams[0]  # 預設抓第一個
    for s in streams:
        if s.name() == stream_name:
            target_stream = s
            break

    print(f"Connecting to {target_stream.name()}...")
    inlet = StreamInlet(target_stream)
    return inlet


if __name__ == "__main__":
    inlet = setup_lsl_inlet()  # 視情況修改你的 Stream 名稱

    thread1 = threading.Thread(target=read_eeg, args=(inlet,), daemon=True)
    thread1.start()

    # 等待一點時間讓 buffer 累積
    print("Buffering...")
    time.sleep(2)

    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        exit(0)
