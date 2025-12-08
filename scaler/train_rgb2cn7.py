#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dataset.py
鼠标框选 ROI，提取中心 5×5 平均 RGB 并写入 train.csv
"""

import os
import cv2
import numpy as np
import csv
import re

# 支持的颜色通道与映射表
COLOR_MAP = {
    'white':      'white',
    'black':      'black',
    'red':        'red',
    'brown':      'ocher',     
    'yellow':     'vineYellow',
    'blue':       'stoneBlue',
    'green':      'stoneGreen',
    'black':      'black'   
}

CSV_HEADER = ['R', 'G', 'B',
              'white', 'stoneGreen', 'stoneBlue',
              'vineYellow', 'red', 'ocher', 'gateWhite']

# 全局变量，用于鼠标回调
g_roi = None
g_done = False

def mouse_cb(event, x, y, flags, param):
    global g_roi, g_done
    img = param['img'].copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        g_roi = [(x, y)]
        g_done = False
    elif event == cv2.EVENT_LBUTTONUP:
        g_roi.append((x, y))
        g_done = True
        cv2.rectangle(img, g_roi[0], g_roi[1], (0, 255, 0), 2)
        cv2.imshow('select', img)

def parse_filename(fname: str):
    """
    从文件名提取两种颜色及比例
    例：Red Black 1_1.jpg  ->  {'red':0.5, 'black':0.5}
    """
    base = os.path.splitext(fname)[0].lower()
    # 用正则提取两个颜色及比例
    m = re.match(r'(\w+)\s+(\w+)\s+(\d+)_(\d+)', base)
    if not m:
        return None
    c1, c2, n1, n2 = m.groups()
    c1, c2 = c1.lower(), c2.lower()
    total = int(n1) + int(n2)
    ratio = {c1: int(n1)/total, c2: int(n2)/total}
    return ratio

def ratio_to_vector(ratio: dict):
    """
    将颜色比例映射到 10 维向量
    """
    vec = np.zeros(len(CSV_HEADER)-3, dtype=np.float32)
    for color, val in ratio.items():
        key = COLOR_MAP.get(color)
        if key and key in CSV_HEADER:
            idx = CSV_HEADER.index(key) - 3   # 前 3 列是 RGB
            vec[idx] = val
    return vec

def process_one_image(img_path):
    global g_roi, g_done
    g_roi = None
    g_done = False

    img = cv2.imread(img_path)
    if img is None:
        print('[WARN] 无法读取', img_path)
        return None
    img_show = img.copy()
    cv2.namedWindow('select', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('select', mouse_cb, {'img': img_show})
    cv2.putText(img_show, 'Drag to select ROI, press any key to confirm',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('select', img_show)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    if not g_done or len(g_roi) != 2:
        print('[WARN] 未框选有效区域，跳过:', img_path)
        return None

    (x1, y1), (x2, y2) = g_roi
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        print('[WARN] ROI 为空，跳过:', img_path)
        return None

    # 取中心 5×5 计算平均 RGB
    h, w = roi.shape[:2]
    cx, cy = w//2, h//2
    half = 2
    patch = roi[max(0, cy-half):min(h, cy+half+1),
                max(0, cx-half):min(w, cx+half+1)]
    mean_bgr = cv2.mean(patch)[:3]
    mean_rgb = np.array(mean_bgr[::-1], dtype=np.float32)

    # 解析文件名
    ratio = parse_filename(os.path.basename(img_path))
    if ratio is None:
        print('[WARN] 文件名格式不符，跳过:', img_path)
        return None
    vec = ratio_to_vector(ratio)
    row = np.concatenate([mean_rgb, vec])
    return row

def main():
    data_dir = 'E:\dm\web_ousu\scaler\data'
    csv_file = 'train.csv'
    if not os.path.isdir(data_dir):
        print('[ERROR] 目录不存在:', data_dir)
        return

    # 读取已有 CSV，避免重复写表头
    write_header = not os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)

        for fname in sorted(os.listdir(data_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png']:
                continue
            full_path = os.path.join(data_dir, fname)
            print('处理:', fname)
            row = process_one_image(full_path)
            if row is not None:
                writer.writerow(row)
                print('  已写入')
            else:
                print('  跳过')

    print('全部完成，结果已保存至:', csv_file)

if __name__ == '__main__':
    main()