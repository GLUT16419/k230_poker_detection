import gc
import os
import aicube
import image
import nncase_runtime as nn
import ujson
import ulab.numpy as np
from libs.Utils import *
from media.media import *
from media.sensor import *
from machine import UART
from machine import FPIOA

OUT_RGB888P_WIDTH = ALIGN_UP(320, 16)
OUT_RGB888P_HEIGH = 240

kmodel_path = "/sdcard/mp_deployment_source/3.kmodel"
config_path = "/sdcard/mp_deployment_source/deploy_config.json"


def two_side_pad_param(input_size, output_size):
    ratio_w = output_size[0] / input_size[0]
    ratio_h = output_size[1] / input_size[1]
    ratio = min(ratio_w, ratio_h)
    new_w = int(ratio * input_size[0])
    new_h = int(ratio * input_size[1])
    dw = (output_size[0] - new_w) / 2
    dh = (output_size[1] - new_h) / 2
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw - 0.1))
    return top, bottom, left, right, ratio


def read_deploy_config(config_path):
    with open(config_path, "r") as json_file:
        try:
            config = ujson.load(json_file)
        except ValueError as e:
            print("JSON parse error:", e)
    return config


def detection():
    print("Loading config...")
    deploy_conf = read_deploy_config(config_path)
    labels = deploy_conf["categories"]
    confidence_threshold = deploy_conf["confidence_threshold"]
    nms_threshold = deploy_conf["nms_threshold"]
    img_size = deploy_conf["img_size"]
    num_classes = deploy_conf["num_classes"]
    nms_option = deploy_conf["nms_option"]
    model_type = deploy_conf["model_type"]
    if model_type == "AnchorBaseDet":
        anchors = deploy_conf["anchors"][0] + deploy_conf["anchors"][1] + deploy_conf["anchors"][2]
    kmodel_frame_size = img_size
    frame_size = [OUT_RGB888P_WIDTH, OUT_RGB888P_HEIGH]
    strides = [8, 16, 32]

    print("Loading model...")
    top, bottom, left, right, ratio = two_side_pad_param(frame_size, kmodel_frame_size)

    print("Setting up UART...")
    fpioa = FPIOA()
    fpioa.set_function(11, fpioa.UART2_TXD)
    fpioa.set_function(12, fpioa.UART2_RXD)
    uart2 = UART(UART.UART2, baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)

    print("Initializing sensor...")
    sensor = Sensor()
    sensor.reset()
    sensor.set_hmirror(False)
    sensor.set_vflip(False)
    sensor.set_framesize(width=OUT_RGB888P_WIDTH, height=OUT_RGB888P_HEIGH, chn=CAM_CHN_ID_2)
    sensor.set_pixformat(PIXEL_FORMAT_RGB_888_PLANAR, chn=CAM_CHN_ID_2)
    MediaManager.init()
    sensor.run()
    print("Sensor ready")

    print("Initializing AI2D...")
    ai2d = nn.ai2d()
    ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
    ai2d.set_pad_param(True, [0, 0, 0, 0, top, bottom, left, right], 0, [114, 114, 114])
    ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
    ai2d_builder = ai2d.build(
        [1, 3, OUT_RGB888P_HEIGH, OUT_RGB888P_WIDTH], [1, 3, kmodel_frame_size[1], kmodel_frame_size[0]]
    )

    print("Initializing KPI...")
    kpu = nn.kpu()
    kpu.load_kmodel(kmodel_path)

    data = np.ones((1, 3, kmodel_frame_size[1], kmodel_frame_size[0]), dtype=np.uint8)
    ai2d_output_tensor = nn.from_numpy(data)

    print("A2: Temporal Filter (Consecutive Frame Detection)...")
    print("Free memory:", gc.mem_free())

    HISTORY_SIZE = 4
    MIN_FRAME_COUNT = 2
    detection_history = []

    frame_count = 0
    try:
        while True:
            frame_count += 1
            if frame_count % 60 == 0:
                print("Frame:", frame_count, "Free memory:", gc.mem_free())
                gc.collect()

            rgb888p_img = sensor.snapshot(chn=CAM_CHN_ID_2)
            if rgb888p_img is None:
                continue

            if rgb888p_img.format() == image.RGBP888:
                ai2d_input = rgb888p_img.to_numpy_ref()
                ai2d_input_tensor = nn.from_numpy(ai2d_input)

                ai2d_builder.run(ai2d_input_tensor, ai2d_output_tensor)
                kpu.set_input_tensor(0, ai2d_output_tensor)
                kpu.run()

                results = []
                for i in range(kpu.outputs_size()):
                    out_data = kpu.get_output_tensor(i)
                    result = out_data.to_numpy()
                    result = result.reshape((result.shape[0] * result.shape[1] * result.shape[2] * result.shape[3]))
                    del out_data
                    results.append(result)

                det_boxes = aicube.anchorbasedet_post_process(
                    results[0],
                    results[1],
                    results[2],
                    kmodel_frame_size,
                    frame_size,
                    strides,
                    num_classes,
                    confidence_threshold,
                    nms_threshold,
                    anchors,
                    nms_option,
                )

                if det_boxes:
                    current_labels = []
                    for det_boxe in det_boxes:
                        label = labels[det_boxe[0]]
                        score = round(det_boxe[1], 2)
                        current_labels.append((label, score))

                    detection_history.append(current_labels)

                    if len(detection_history) > HISTORY_SIZE:
                        detection_history.pop(0)

                    if len(detection_history) >= MIN_FRAME_COUNT:
                        label_scores = {}
                        for hist_labels in detection_history[-MIN_FRAME_COUNT:]:
                            for label, score in hist_labels:
                                if label not in label_scores:
                                    label_scores[label] = []
                                label_scores[label].append(score)

                        uart_data = ""
                        for label, scores in label_scores.items():
                            avg_score = sum(scores) / len(scores)
                            uart_data += "{}:{:.2f},".format(label, avg_score)

                        print("Stable detection:", uart_data)
                        uart2.write((uart_data + "\n").encode("utf-8"))
                        detection_history.clear()

                del ai2d_input
                del ai2d_input_tensor
                del results
                del det_boxes

            del rgb888p_img
            gc.collect()

    except Exception as e:
        print("Error:", e)
        import sys
        sys.print_exception(e)

    del ai2d_output_tensor
    sensor.stop()
    MediaManager.deinit()
    uart2.deinit()
    gc.collect()
    nn.shrink_memory_pool()
    print("Detection ended")
    return 0


if __name__ == "__main__":
    detection()
