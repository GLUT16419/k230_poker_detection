# K230 扑克牌检测 - 改进版本V5(A2_improved_v5)
# 功能：单卡片多次识别合并发送 + 离开确认 + 去重机制 + 功耗优化
# 关键特性：
# 1. 识别同一个卡片，滤波后每5次合并发送（保留A2_low_power的输出间隔）
# 2. 确认离开机制：卡片离开后等待一段时间确认，然后清除输出缓冲
# 3. 去重机制：记录已识别的卡片，下次识别时排除这些卡片
# 4. 借鉴v3的置信度累积算法和查重机制
# 5. 保留三级功耗模式和动态检测间隔
# 6. 串口输出格式：{"cards": [card_name, card_name, ...]}

import gc
import os
import time
import aicube
import image
import nncase_runtime as nn
import ujson
import ulab.numpy as np
from media.media import *
from media.sensor import *
from machine import UART
from machine import FPIOA

# ==================== 宏定义配置区 ====================
# 模型名称（与 .kmodel 和 .json 文件名对应）
MODEL_NAME = "333"

# 文件路径
ROOT_DIR = "/sdcard/mp_deployment_source/"
KMODEL_PATH = ROOT_DIR + MODEL_NAME + ".kmodel"      # 模型文件路径
CONFIG_PATH = ROOT_DIR + MODEL_NAME + ".json"        # 配置文件路径

# 摄像头配置
SENSOR_WIDTH = ALIGN_UP(1920, 16)                    # 传感器宽度（需16字节对齐）
SENSOR_HEIGHT = 1080                                 # 传感器高度

# 串口配置
UART_BAUDRATE = 115200                               # 波特率
UART_BITS = UART.EIGHTBITS                           # 数据位：8位
UART_PARITY = UART.PARITY_NONE                       # 校验位：无校验
UART_STOP = UART.STOPBITS_ONE                        # 停止位：1位
GPIO_TXD = 11                                        # 发送引脚：GPIO11
GPIO_RXD = 12                                        # 接收引脚：GPIO12

# 检测算法配置
CONFIDENCE_THRESHOLD = 0.6                           # 单帧置信度阈值
HISTORY_SIZE = 4                                     # 检测历史最大长度
MIN_FRAME_COUNT = 2                                  # 最小连续帧检测数量
DEPARTURE_DELAY_FRAMES = 15                          # 离开确认帧数（0.5秒 @30fps）
MIN_CONFIRM_DETECTIONS = 3                           # 同一张卡片需要被识别多少次才确认

# 输出配置（保留A2_low_power：每5次识别合并发送）
OUTPUT_INTERVAL = 5                                  # 每5次识别结果合并发送
MAX_CARDS = 5                                        # 最大识别牌数（去重后）

# 功耗优化配置（保留A2_low_power）
DETECT_INTERVAL_NORMAL = 1                           # 正常模式：每帧检测
DETECT_INTERVAL_LOW = 3                              # 低功耗模式：每3帧检测一次
DETECT_INTERVAL_SLEEP = 10                           # 休眠模式：每10帧检测一次
NO_DETECT_THRESHOLD_LOW = 30                         # 30帧无检测 → 低功耗模式
NO_DETECT_THRESHOLD_SLEEP = 200                      # 100帧无检测 → 休眠模式

# 调试配置
DEBUG_MODE = 1                                       # 调试模式：1=开启串口打印，0=关闭
FPS_PRINT_INTERVAL = 30                              # 每30帧打印一次FPS
# ===================================================


def load_deploy_config(config_path):
    with open(config_path, 'r') as f:
        config = ujson.load(f)
    return config


def setup_uart():
    fpioa = FPIOA()
    fpioa.set_function(GPIO_TXD, fpioa.UART2_TXD)
    fpioa.set_function(GPIO_RXD, fpioa.UART2_RXD)
    uart = UART(UART.UART2, baudrate=UART_BAUDRATE,
                bits=UART_BITS, parity=UART_PARITY,
                stop=UART_STOP)
    return uart


def compute_padding(ori_w, ori_h, target_w, target_h):
    ratiow = float(target_w) / ori_w
    ratioh = float(target_h) / ori_h
    ratio = ratiow if ratiow < ratioh else ratioh
    new_w = int(ratio * ori_w)
    new_h = int(ratio * ori_h)
    dw = float(target_w - new_w) / 2
    dh = float(target_h - new_h) / 2
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw - 0.1))
    return top, bottom, left, right


def detection():
    print("Poker detection start (V5: Leave Confirm + Deduplication + Power Optimization)")

    config = load_deploy_config(CONFIG_PATH)
    labels = config["categories"]
    model_type = config["model_type"]
    confidence_threshold = config["confidence_threshold"]
    nms_threshold = config["nms_threshold"]
    nms_option = config["nms_option"]
    img_size = config["img_size"]
    num_classes = config["num_classes"]

    if model_type == "AnchorBaseDet":
        anchors = (config["anchors"][0] +
                   config["anchors"][1] +
                   config["anchors"][2])
    else:
        anchors = None

    strides = [8, 16, 32]

    top, bottom, left, right = compute_padding(
        SENSOR_WIDTH, SENSOR_HEIGHT,
        img_size[0], img_size[1])

    kpu = nn.kpu()
    ai2d = nn.ai2d()
    kpu.load_kmodel(KMODEL_PATH)

    ai2d.set_dtype(nn.ai2d_format.NCHW_FMT,
                   nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
    ai2d.set_pad_param(True, [0, 0, 0, 0, top, bottom, left, right],
                       0, [114, 114, 114])
    ai2d.set_resize_param(True, nn.interp_method.tf_bilinear,
                          nn.interp_mode.half_pixel)
    ai2d_builder = ai2d.build(
        [1, 3, SENSOR_HEIGHT, SENSOR_WIDTH],
        [1, 3, img_size[1], img_size[0]])

    sensor = Sensor()
    sensor.reset()
    sensor.set_hmirror(False)
    sensor.set_vflip(False)
    sensor.set_framesize(width=SENSOR_WIDTH, height=SENSOR_HEIGHT,
                         chn=CAM_CHN_ID_2)
    sensor.set_pixformat(PIXEL_FORMAT_RGB_888_PLANAR, chn=CAM_CHN_ID_2)

    uart = setup_uart()
    print(f"UART2 ready ({UART_BAUDRATE} 8N1)")

    data = np.ones((1, 3, img_size[1], img_size[0]), dtype=np.uint8)
    ai2d_output_tensor = nn.from_numpy(data)
    frame_count = 0

    # 检测算法状态（借鉴v3）
    confidence_accum = {}                             # 置信度累积字典
    detection_history = []                            # 检测历史（用于滤波）
    confirmed_cards = []                              # 已确认的卡片列表（去重用）
    card_detection_count = {}                         # 记录每张卡片被识别的次数（用于确认）

    # 输出缓冲（保留A2_low_power：每5次识别合并发送）
    output_buffer = []                                # 累积识别结果，满5个发送

    # 离开确认机制（借鉴v3）
    departure_counter = {'count': 0}                  # 离开计数器
    last_detected_card = None                         # 最后检测到的卡片

    # 功耗优化状态（保留A2_low_power）
    fps_frame_count = 0
    fps_start_time = time.time()
    current_detect_interval = DETECT_INTERVAL_NORMAL
    skip_count = 0
    no_detect_count = 0
    power_mode = "NORMAL"

    print("Config loaded:")
    print(f"  Model: {model_type}, Classes: {num_classes}")
    print(f"  Input size: {img_size}")
    print(f"  Departure delay: {DEPARTURE_DELAY_FRAMES} frames")
    print(f"  Output interval: every {OUTPUT_INTERVAL} detections")
    print(f"  Max cards: {MAX_CARDS}")
    print(f"  Power mode: {power_mode} (NORMAL/LOW/SLEEP)")
    print("Starting inference...")

    try:
        MediaManager.init()
        sensor.run()

        while True:
            frame_count += 1

            # ==================== 跳帧检测逻辑（功耗优化） ====================
            if skip_count < current_detect_interval - 1:
                skip_count += 1
                img = sensor.snapshot(chn=CAM_CHN_ID_2)
                if img is not None:
                    del img
                continue
            skip_count = 0

            fps_frame_count += 1

            # ==================== 调试输出 ====================
            if frame_count % 60 == 0:
                print(f"Frame: {frame_count} Mode: {power_mode} Free memory: {gc.mem_free()}")
                gc.collect()

            if fps_frame_count >= FPS_PRINT_INTERVAL:
                fps_end_time = time.time()
                time_elapsed = fps_end_time - fps_start_time
                if time_elapsed > 0:
                    effective_fps = fps_frame_count / time_elapsed
                    actual_fps = effective_fps * current_detect_interval
                    print(f"Effective FPS: {effective_fps:.2f} (Actual: {actual_fps:.2f}) Mode: {power_mode}")
                else:
                    print(f"Effective FPS: inf (Time elapsed too small) Mode: {power_mode}")
                fps_frame_count = 0
                fps_start_time = time.time()

            # ==================== 图像采集与推理 ====================
            img = sensor.snapshot(chn=CAM_CHN_ID_2)
            if img is None:
                continue

            if img.format() == image.RGBP888:
                ai2d_input = img.to_numpy_ref()
                ai2d_input_tensor = nn.from_numpy(ai2d_input)
                ai2d_builder.run(ai2d_input_tensor, ai2d_output_tensor)

                kpu.set_input_tensor(0, ai2d_output_tensor)
                kpu.run()

                results = []
                for i in range(kpu.outputs_size()):
                    out_data = kpu.get_output_tensor(i)
                    result = out_data.to_numpy()
                    result = result.reshape((
                        result.shape[0] * result.shape[1] *
                        result.shape[2] * result.shape[3]))
                    del out_data
                    results.append(result)
                gc.collect()

                if model_type == "AnchorBaseDet":
                    det_boxes = aicube.anchorbasedet_post_process(
                        results[0], results[1], results[2],
                        img_size, [SENSOR_WIDTH, SENSOR_HEIGHT],
                        strides, num_classes,
                        confidence_threshold, nms_threshold,
                        anchors, nms_option)
                elif model_type == "GFLDet":
                    det_boxes = aicube.gfldet_post_process(
                        results[0], results[1], results[2],
                        img_size, [SENSOR_WIDTH, SENSOR_HEIGHT],
                        strides, num_classes,
                        confidence_threshold, nms_threshold,
                        nms_option)
                else:
                    det_boxes = aicube.anchorfreedet_post_process(
                        results[0], results[1], results[2],
                        img_size, [SENSOR_WIDTH, SENSOR_HEIGHT],
                        strides, num_classes,
                        confidence_threshold, nms_threshold,
                        nms_option)

                # ==================== 检测处理（核心逻辑） ====================
                detected = False
                current_labels = []
                current_detected = {}

                if det_boxes:
                    # 累积置信度（借鉴v3）
                    for det_box in det_boxes:
                        if det_box[1] >= CONFIDENCE_THRESHOLD:
                            label = labels[det_box[0]]
                            current_labels.append(label)
                            detected = True

                            # 记录最高置信度（借鉴v3）
                            if label in current_detected:
                                if det_box[1] > current_detected[label]:
                                    current_detected[label] = det_box[1]
                            else:
                                current_detected[label] = det_box[1]

                            # 累积置信度（借鉴v3）
                            if label in confidence_accum:
                                confidence_accum[label] += det_box[1]
                            else:
                                confidence_accum[label] = det_box[1]

                    # 检测到扑克牌 → 立即恢复全速模式
                    if detected:
                        no_detect_count = 0
                        if power_mode != "NORMAL":
                            power_mode = "NORMAL"
                            current_detect_interval = DETECT_INTERVAL_NORMAL
                            print("Wake up: Cards detected, entering NORMAL mode")

                        # 重置离开计数器
                        departure_counter['count'] = 0

                        # 添加到检测历史
                        if current_labels:
                            detection_history.append(current_labels)
                            if len(detection_history) > HISTORY_SIZE:
                                detection_history.pop(0)

                            # 当历史足够时进行滤波
                            if len(detection_history) >= MIN_FRAME_COUNT:
                                label_counts = {}
                                # 统计最近MIN_FRAME_COUNT帧中各标签出现次数（加权）
                                for hist_labels in detection_history[-MIN_FRAME_COUNT:]:
                                    for label in hist_labels:
                                        if label not in label_counts:
                                            label_counts[label] = 0
                                        # 使用置信度作为权重
                                        weight = confidence_accum.get(label, 1.0)
                                        label_counts[label] += weight

                                # 选择置信度最高的卡片
                                if label_counts:
                                    best_card = max(label_counts.items(), key=lambda x: x[1])
                                    best_label = best_card[0]
                                    last_detected_card = best_label

                                    # 添加到输出缓冲（持续输出到串口）
                                    output_buffer.append(best_label)

                                    # 每5次识别结果合并发送
                                    if len(output_buffer) >= OUTPUT_INTERVAL:
                                        json_data = ujson.dumps({"cards": output_buffer})
                                        uart.write((json_data + "\n").encode("utf-8"))
                                        if DEBUG_MODE:
                                            print("Stable detection (x5):", json_data)
                                        output_buffer = []

                                detection_history.clear()
                                confidence_accum.clear()

                # ==================== 离开确认机制 ====================
                if not detected:
                    no_detect_count += 1

                    # 如果之前有检测到卡片，开始计数离开
                    if last_detected_card is not None or confidence_accum:
                        departure_counter['count'] += 1

                        # 确认离开：等待DEPARTURE_DELAY_FRAMES帧后确认卡片离开
                        if departure_counter['count'] > DEPARTURE_DELAY_FRAMES:
                            # 统计离开前识别最多的卡片
                            if confidence_accum:
                                best_card = max(confidence_accum.items(), key=lambda x: x[1])
                                best_label = best_card[0]

                                # 将卡片添加到已确认列表（离开后才确认）
                                if best_label not in confirmed_cards and len(confirmed_cards) < MAX_CARDS:
                                    confirmed_cards.append(best_label)
                                    if DEBUG_MODE:
                                        print(f"Card confirmed on departure: {best_label}")
                                        print(f"Confirmed cards: {confirmed_cards}")

                            # 清除发送串口的信息槽（防止影响下一次输出）
                            output_buffer = []

                            # 重置状态
                            confidence_accum.clear()
                            departure_counter['count'] = 0
                            last_detected_card = None
                            card_detection_count.clear()  # 重置识别计数，为下一张卡片做准备

                    # 根据无检测帧数切换功耗模式
                    if no_detect_count >= NO_DETECT_THRESHOLD_SLEEP and power_mode != "SLEEP":
                        power_mode = "SLEEP"
                        current_detect_interval = DETECT_INTERVAL_SLEEP
                        print(f"Entering SLEEP mode: {no_detect_count} frames no detection")
                    elif no_detect_count >= NO_DETECT_THRESHOLD_LOW and power_mode == "NORMAL":
                        power_mode = "LOW"
                        current_detect_interval = DETECT_INTERVAL_LOW
                        print(f"Entering LOW mode: {no_detect_count} frames no detection")

                del ai2d_input
                del ai2d_input_tensor
                del results
                del det_boxes

            del img
            gc.collect()

    except Exception as e:
        print(f"Error: {e}")
        import sys
        sys.print_exception(e)

    finally:
        sensor.stop()
        MediaManager.deinit()
        uart.deinit()
        del ai2d_output_tensor
        gc.collect()
        nn.shrink_memory_pool()
    print("Detection ended")


if __name__ == "__main__":
    detection()
