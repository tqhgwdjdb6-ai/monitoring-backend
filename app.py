# app.py - 监控平台后端API服务
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import datetime
import random
import json
import os
import re
import struct
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq

# 初始化Flask应用
app = Flask(__name__)

# 配置CORS，允许所有来源（生产环境应该限制）
CORS(app, resources={r"/api/*": {"origins": "*"}})

# 从环境变量获取配置，如果不存在则使用默认值
BASE_URL = os.environ.get('BASE_URL', 'http://58.57.159.186:30200')
SAMPLE_RATE = int(os.environ.get('SAMPLE_RATE', 50))
FRAME_HEADER = b'\x55\xaa'
FRAME_LEN = 160
WINDOW_SIZE = 10 * 60 * SAMPLE_RATE  # 10分钟窗口大小


# ======================================
# 健康检查端点
# ======================================
@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "service": "Monitoring Platform API",
        "version": "1.0.0",
        "timestamp": datetime.datetime.now().isoformat()
    })


@app.route('/api/test', methods=['GET'])
def test_api():
    """测试接口"""
    return jsonify({
        "message": "API is working!",
        "base_url": BASE_URL,
        "endpoints": [
            "POST /api/imu_platform_swing - IMU平台晃动分析",
            "POST /api/wind_wave_data - 风浪数据获取"
        ]
    })


# ======================================
# IMU数据处理函数（从getimudata.py提取）
# ======================================
def get_gnss_data_names(year, month, day, hour):
    """获取GNSS数据文件名列表"""
    try:
        r = requests.post(
            f"{BASE_URL}/getdata/getgnssdatanames",
            json={"year": year, "month": month, "day": day, "hour": hour},
            timeout=10
        )
        r.raise_for_status()
        files = r.json().get("files", [])
        return files
    except Exception as e:
        print(f"❌ 查询文件名失败: {e}")
        return []


def get_bin_bytes(sdt):
    """获取二进制文件内容"""
    try:
        r = requests.get(f"{BASE_URL}/getdata/getGnssData/{sdt}", timeout=10)
        if r.status_code == 200:
            return r.content
        elif r.status_code == 404:
            print(f"❌ 文件不存在: {sdt}")
            return None
        else:
            print(f"❌ 获取失败，状态码: {r.status_code}")
            return None
    except Exception as e:
        print(f"❌ 获取文件失败: {e}")
        return None


def parse_frame(data: bytes):
    """解析单帧数据"""
    frame_data = {
        'timestamp': struct.unpack_from('<I', data, 3)[0],
        'week': struct.unpack_from('<H', data, 7)[0],
        'accX_m_s2': struct.unpack_from('<i', data, 27)[0] * 0.000001,
        'accY_m_s2': struct.unpack_from('<i', data, 31)[0] * 0.000001,
        'accZ_m_s2': struct.unpack_from('<i', data, 35)[0] * 0.000001,
        'gyroX_rad_s': struct.unpack_from('<i', data, 39)[0] * 0.000001,
        'gyroY_rad_s': struct.unpack_from('<i', data, 43)[0] * 0.000001,
        'gyroZ_rad_s': struct.unpack_from('<i', data, 47)[0] * 0.000001,
        'roll_deg': struct.unpack_from('<i', data, 51)[0] * 0.000001,
        'pitch_deg': struct.unpack_from('<i', data, 55)[0] * 0.000001,
        'yaw_deg': struct.unpack_from('<i', data, 59)[0] * 0.000001,
        'latitude_deg': struct.unpack_from('<i', data, 63)[0] * 0.000001,
        'longitude_deg': struct.unpack_from('<i', data, 67)[0] * 0.000001,
        'altitude_m': struct.unpack_from('<i', data, 71)[0] * 0.000001,
        'velocityNorth_m_s': struct.unpack_from('<i', data, 75)[0] * 0.000001,
        'velocityEast_m_s': struct.unpack_from('<i', data, 79)[0] * 0.000001,
        'velocityUp_m_s': struct.unpack_from('<i', data, 83)[0] * 0.000001,
        'gnss_status': data[87],
        'satellite_num': data[88],
        'temperature_C': struct.unpack_from('<h', data, 89)[0] * 0.01,
        'pressure_hPa': struct.unpack_from('<I', data, 91)[0] * 0.001,
    }
    return frame_data


def parse_bin_bytes(content: bytes, base_time: datetime.datetime):
    """解析整个二进制数据"""
    frames = []
    i = 0
    frame_count = 0

    while i < len(content) - FRAME_LEN:
        if content[i:i + 2] == FRAME_HEADER:
            frame_data = parse_frame(content[i:i + FRAME_LEN])

            # 计算每个数据点的时间戳
            time_offset = frame_count * (1.0 / SAMPLE_RATE)
            frame_time = base_time + datetime.timedelta(seconds=time_offset)
            frame_data['time_str'] = frame_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            frame_data['timestamp_seconds'] = frame_time.timestamp()

            frames.append(frame_data)
            i += FRAME_LEN
            frame_count += 1
        else:
            i += 1

    return pd.DataFrame(frames)


def extract_timestamp_from_filename(filename):
    """从文件名提取时间戳"""
    match = re.search(r'data_(\d{12})\.bin', filename)
    if match:
        timestamp = match.group(1)
        if len(timestamp) == 12 and timestamp.isdigit():
            return timestamp
    return None


def acceleration_to_displacement(acceleration, sample_rate=SAMPLE_RATE):
    """频域双重积分：加速度 -> 位移"""
    n = len(acceleration)
    if n == 0 or np.std(acceleration) < 1e-10:
        return np.zeros_like(acceleration)

    # 去均值
    acceleration = acceleration - np.mean(acceleration)

    # 加窗处理
    window = np.hanning(n)
    acceleration_windowed = acceleration * window

    # FFT
    fft_acc = np.fft.fft(acceleration_windowed)
    frequencies = np.fft.fftfreq(n, 1 / sample_rate)
    omega = 2 * np.pi * frequencies

    # 设置低频截止
    min_freq = 0.1  # Hz
    omega_threshold = 2 * np.pi * min_freq
    omega_sq = np.zeros_like(omega, dtype=complex)

    for i, w in enumerate(omega):
        omega_sq[i] = -omega_threshold ** 2 if abs(w) < omega_threshold else -w ** 2

    # 频域积分
    fft_disp = fft_acc / omega_sq
    fft_disp[0] = 0  # 去除DC分量

    # 逆FFT
    displacement = np.real(np.fft.ifft(fft_disp))

    # 窗函数补偿
    window_compensation = np.mean(window)
    if window_compensation > 0:
        displacement /= window_compensation

    return displacement


def gaussian(x, a, mu, sigma):
    """高斯函数"""
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def gaussian_fit_displacement(displacement):
    """高斯拟合"""
    if len(displacement) == 0 or np.std(displacement) < 1e-10:
        return 0.0, False

    # 计算直方图
    hist, bin_edges = np.histogram(displacement, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    try:
        initial_guess = [np.max(hist), np.mean(displacement), np.std(displacement)]
        popt, _ = curve_fit(gaussian, bin_centers, hist, p0=initial_guess, maxfev=5000)
        _, _, sigma_fit = popt
        return float(sigma_fit), True
    except:
        return float(np.std(displacement)), False


def extract_dominant_frequency(acceleration, sample_rate=SAMPLE_RATE):
    """提取加速度数据的主频率和周期"""
    n = len(acceleration)
    if n < 10 or np.std(acceleration) < 1e-10:
        return 0.0, 0.0

    # 去均值
    acceleration = acceleration - np.mean(acceleration)

    # 加窗
    window = np.hanning(n)
    acceleration_windowed = acceleration * window

    # FFT
    fft_values = np.fft.fft(acceleration_windowed)
    frequencies = np.fft.fftfreq(n, 1 / sample_rate)

    # 取幅度谱
    magnitude = np.abs(fft_values)

    # 只考虑正频率部分
    positive_freq_mask = frequencies > 0
    positive_freqs = frequencies[positive_freq_mask]
    positive_magnitude = magnitude[positive_freq_mask]

    if len(positive_freqs) == 0:
        return 0.0, 0.0

    # 忽略直流和极低频成分
    min_freq_threshold = 0.1  # Hz
    valid_mask = positive_freqs > min_freq_threshold
    if not np.any(valid_mask):
        return 0.0, 0.0

    valid_freqs = positive_freqs[valid_mask]
    valid_magnitude = positive_magnitude[valid_mask]

    # 找到主频率（幅度最大的频率）
    dominant_idx = np.argmax(valid_magnitude)
    dominant_freq = valid_freqs[dominant_idx]

    # 计算周期
    period = 1.0 / dominant_freq if dominant_freq > 0 else 0.0

    return float(dominant_freq), float(period)


def process_window_data(window_df):
    """处理单个10分钟窗口的数据"""
    if len(window_df) == 0:
        return None

    # 窗口开始时间
    window_start_time = window_df.iloc[0]['time_str']

    # 获取加速度数据
    acc_east = window_df['accY_m_s2'].values  # 东向加速度
    acc_north = window_df['accX_m_s2'].values  # 北向加速度
    acc_up = window_df['accZ_m_s2'].values  # 天向加速度

    # 计算位移
    disp_east = acceleration_to_displacement(acc_east)
    disp_north = acceleration_to_displacement(acc_north)
    disp_up = acceleration_to_displacement(acc_up)

    # 高斯拟合得到晃动位移（标准差）
    sigma_east, _ = gaussian_fit_displacement(disp_east)
    sigma_north, _ = gaussian_fit_displacement(disp_north)
    sigma_up, _ = gaussian_fit_displacement(disp_up)

    # 提取主频率和周期
    freq_east, period_east = extract_dominant_frequency(acc_east)
    freq_north, period_north = extract_dominant_frequency(acc_north)
    freq_up, period_up = extract_dominant_frequency(acc_up)

    # 构建结果字典
    result = {
        "window_start_time": window_start_time,
        "swing_displacement": {
            "east": round(sigma_east, 6),
            "north": round(sigma_north, 6),
            "up": round(sigma_up, 6)
        },
        "dominant_frequency": {
            "east": round(freq_east, 4),
            "north": round(freq_north, 4),
            "up": round(freq_up, 4)
        },
        "swing_period": {
            "east": round(period_east, 2),
            "north": round(period_north, 2),
            "up": round(period_up, 2)
        },
        "window_size": len(window_df),
        "sample_rate": SAMPLE_RATE
    }

    return result


def process_imu_data(st1: str, st2: str, classic=None):
    """处理IMU数据，按10分钟窗口分析平台晃动"""
    dt_start = datetime.datetime.strptime(st1, "%Y%m%d%H%M")
    dt_end = datetime.datetime.strptime(st2, "%Y%m%d%H%M")

    print(f"开始处理IMU数据，时间范围: {dt_start} 到 {dt_end}")
    print(f"站点参数 classic: {classic}")

    # 收集所有文件信息
    all_files_info = []
    current_hour = dt_start.replace(minute=0, second=0, microsecond=0)
    end_hour = dt_end.replace(minute=0, second=0, microsecond=0)

    while current_hour <= end_hour:
        year, month, day, hour = current_hour.year, current_hour.month, current_hour.day, current_hour.hour
        files = get_gnss_data_names(year, month, day, hour)

        print(f"  小时 {current_hour} 找到 {len(files)} 个文件")

        for filename in files:
            sdt = extract_timestamp_from_filename(filename)
            if sdt:
                try:
                    file_dt = datetime.datetime.strptime(sdt, "%Y%m%d%H%M")
                    if dt_start <= file_dt <= dt_end:
                        all_files_info.append({
                            "filename": filename,
                            "sdt": sdt,
                            "file_dt": file_dt
                        })
                        print(f"    匹配文件: {filename}, 时间: {sdt}")
                except Exception as e:
                    print(f"❌ 解析文件时间失败: {filename}, 错误: {e}")

        current_hour += datetime.timedelta(hours=1)

    # 按时间排序
    all_files_info.sort(key=lambda x: x["file_dt"])
    print(f"总共找到 {len(all_files_info)} 个文件需要处理")

    if not all_files_info:
        print("❌ 没有在指定时间范围内找到任何文件")
        return []

    # 合并所有数据
    all_data_frames = []

    for file_info in all_files_info:
        print(f"处理文件: {file_info['filename']}")
        content = get_bin_bytes(file_info["sdt"])
        if content:
            try:
                # 解析文件数据，传入文件时间作为基准时间
                file_dt = file_info["file_dt"]
                df = parse_bin_bytes(content, file_dt)

                if not df.empty:
                    all_data_frames.append(df)
                    print(f"✓ 成功解析文件: {file_info['filename']}, 数据点数: {len(df)}")
                else:
                    print(f"⚠️ 文件解析后无数据: {file_info['filename']}")
            except Exception as e:
                print(f"❌ 解析文件失败: {file_info['filename']}, 错误: {e}")
        else:
            print(f"❌ 无法获取文件内容: {file_info['filename']}")

    if not all_data_frames:
        print("❌ 没有成功解析任何数据")
        return []

    # 合并所有数据
    combined_df = pd.concat(all_data_frames, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp_seconds')

    print(f"合并后总数据点数: {len(combined_df)}")
    print(f"数据时间范围: {combined_df.iloc[0]['time_str']} 到 {combined_df.iloc[-1]['time_str']}")

    # 按10分钟窗口处理数据
    window_results = []
    window_size_samples = WINDOW_SIZE

    for i in range(0, len(combined_df), window_size_samples):
        end_idx = min(i + window_size_samples, len(combined_df))
        window_df = combined_df.iloc[i:end_idx]

        # 确保窗口有足够数据（至少1分钟数据）
        if len(window_df) >= 60 * SAMPLE_RATE:
            result = process_window_data(window_df)
            if result:
                window_results.append(result)
                print(f"✓ 处理窗口 {i // window_size_samples + 1}, 开始时间: {result['window_start_time']}")

    print(f"共处理 {len(window_results)} 个10分钟窗口")
    return window_results


# ======================================
# IMU API接口
# ======================================
@app.route("/api/imu_platform_swing", methods=["POST"])
def imu_platform_swing():
    """IMU平台晃动分析接口"""
    try:
        payload = request.json or {}
        st1 = payload.get("st1")
        st2 = payload.get("st2")
        classic = payload.get("classic")

        print("📡 接收到IMU平台晃动分析请求:")
        print(f"   st1 (起始时间): {st1}")
        print(f"   st2 (结束时间): {st2}")
        print(f"   classic (站点): {classic}")

        if not (st1 and st2):
            return jsonify({"error": "缺少参数 st1, st2"}), 400

        # 验证时间格式
        try:
            datetime.datetime.strptime(st1, "%Y%m%d%H%M")
            datetime.datetime.strptime(st2, "%Y%m%d%H%M")
        except ValueError as e:
            return jsonify({"error": "时间格式错误，应为 YYYYMMDDHHMM"}), 400

        # 处理数据
        results = process_imu_data(st1, st2, classic)

        return jsonify({
            "status": "success",
            "parameters": {
                "start_time": st1,
                "end_time": st2,
                "classic": classic,
                "sample_rate": SAMPLE_RATE,
                "window_size_minutes": 10
            },
            "total_windows": len(results),
            "data": results
        })

    except Exception as e:
        print(f"❌ 接口处理失败: {e}")
        return jsonify({"error": f"处理失败: {str(e)}"}), 500


# ======================================
# 风浪数据处理函数（从getwindwavedata.py提取）
# ======================================
def generate_mock_wind_wave_data(st1, st2, dataname):
    """生成模拟风浪数据"""
    try:
        start_dt = datetime.datetime.strptime(st1, "%Y%m%d%H%M")
        end_dt = datetime.datetime.strptime(st2, "%Y%m%d%H%M")

        data = []
        current_dt = start_dt

        while current_dt <= end_dt:
            if dataname == "wind":
                data.append({
                    "sdt": current_dt.strftime("%Y%m%d%H%M"),
                    "df": 5 + random.random() * 10,  # 风速 5-15 m/s
                    "wd": random.random() * 360,  # 风向 0-360度
                    "ws": 5 + random.random() * 10  # 风速备用
                })
            elif dataname == "wave":
                data.append({
                    "sdt": current_dt.strftime("%Y%m%d%H%M"),
                    "avgH": 0.5 + random.random() * 2,  # 平均浪高 0.5-2.5 m
                    "maxH": 1 + random.random() * 3  # 最大浪高 1-4 m
                })
            else:
                # 其他类型数据
                data.append({
                    "sdt": current_dt.strftime("%Y%m%d%H%M"),
                    "value": random.random() * 100
                })

            # 增加1小时
            current_dt += datetime.timedelta(hours=1)

        print(f"  生成模拟 {dataname} 数据: {len(data)} 条记录")
        return data
    except Exception as e:
        print(f"❌ 生成模拟数据错误: {e}")
        return []


def get_wind_wave_data(st1: str, st2: str, classic: int, dataname: str):
    """
    获取风浪数据
    Args:
        st1: 开始时间 (YYYYMMDDHHMM)
        st2: 结束时间 (YYYYMMDDHHMM)
        classic: 数据类型分类
        dataname: 数据名称 (wind/wave)
    """
    print(f"\n===== 获取 {dataname} 数据 =====")

    try:
        url = f"{BASE_URL}/getdata/getwindwavedata"
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0'
        }

        payload = {
            "sdt1": st1,
            "sdt2": st2,
            "classic": classic,
            "dataname": dataname
        }

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=15
        )

        if response.status_code == 200:
            try:
                response_data = response.json()
                data = response_data.get("data", [])
                print(f"  ✓ {dataname} 数据获取成功: {len(data)} 条记录")
                return {
                    "status": "success",
                    "source": "api",
                    "count": len(data),
                    "data": data
                }
            except json.JSONDecodeError as e:
                print(f"  ❌ 解析JSON响应失败: {e}")

                # 使用模拟数据
                mock_data = generate_mock_wind_wave_data(st1, st2, dataname)
                return {
                    "status": "warning",
                    "source": "mock",
                    "count": len(mock_data),
                    "data": mock_data
                }
        else:
            print(f"  ❌ HTTP错误: {response.status_code}")

            # 使用模拟数据作为备用
            mock_data = generate_mock_wind_wave_data(st1, st2, dataname)
            return {
                "status": "warning",
                "source": "mock",
                "count": len(mock_data),
                "data": mock_data
            }

    except requests.exceptions.Timeout:
        print(f"  ❌ 请求超时: {dataname}")
        mock_data = generate_mock_wind_wave_data(st1, st2, dataname)
        return {
            "status": "warning",
            "source": "mock",
            "count": len(mock_data),
            "data": mock_data
        }

    except requests.exceptions.ConnectionError:
        print(f"  ❌ 连接错误: {dataname}")
        mock_data = generate_mock_wind_wave_data(st1, st2, dataname)
        return {
            "status": "warning",
            "source": "mock",
            "count": len(mock_data),
            "data": mock_data
        }

    except Exception as e:
        print(f"  ❌ 获取 {dataname} 数据失败: {e}")
        mock_data = generate_mock_wind_wave_data(st1, st2, dataname)
        return {
            "status": "error",
            "source": "mock",
            "count": len(mock_data),
            "data": mock_data
        }


# ======================================
# 风浪数据API接口
# ======================================
@app.route("/api/wind_wave_data", methods=["POST"])
def wind_wave_data():
    """获取风浪数据"""
    try:
        payload = request.json or {}
        print("\n=== /api/wind_wave_data 接口调用 ===")
        print(f"参数: st1={payload.get('st1')}, st2={payload.get('st2')}, classic={payload.get('classic')}")

        st1 = payload.get("st1")
        st2 = payload.get("st2")
        classic = payload.get("classic")

        if not (st1 and st2 and classic):
            return jsonify({"error": "缺少参数: st1, st2, classic"}), 400

        # 验证时间格式
        try:
            datetime.datetime.strptime(st1, "%Y%m%d%H%M")
            datetime.datetime.strptime(st2, "%Y%m%d%H%M")
        except ValueError:
            return jsonify({"error": "时间格式错误，应为 YYYYMMDDHHMM"}), 400

        classic = int(classic)

        # 获取风浪数据
        wind_result = get_wind_wave_data(st1, st2, classic, "wind")
        wave_result = get_wind_wave_data(st1, st2, classic, "wave")

        print(f"风数据状态: {wind_result.get('status')}, 数量: {wind_result.get('count')}")
        print(f"浪数据状态: {wave_result.get('status')}, 数量: {wave_result.get('count')}")

        # 组合结果
        response_data = {
            "status": "success",
            "wind": wind_result,
            "wave": wave_result,
            "request": {
                "st1": st1,
                "st2": st2,
                "classic": classic
            }
        }

        return jsonify(response_data)

    except Exception as e:
        error_msg = str(e)
        print(f"API处理错误: {error_msg}")
        return jsonify({
            "status": "error",
            "message": f"服务器内部错误: {error_msg}"
        }), 500


# ======================================
# 主程序入口
# ======================================
if __name__ == "__main__":
    # 获取端口号，Render会自动设置PORT环境变量
    port = int(os.environ.get("PORT", 5000))

    print("=" * 50)
    print("🚀 监控平台后端API服务启动")
    print(f"   端口: {port}")
    print(f"   BASE_URL: {BASE_URL}")
    print(f"   接口:")
    print(f"     1. GET /health - 健康检查")
    print(f"     2. POST /api/imu_platform_swing - IMU平台晃动分析")
    print(f"     3. POST /api/wind_wave_data - 风浪数据获取")
    print("=" * 50)

    # 在Render上运行时使用0.0.0.0
    app.run(host="0.0.0.0", port=port, debug=False)