#!/bin/bash
# 云仿真环境初始化脚本 - 安装依赖
set -e

echo "=== 安装系统依赖 ==="
sudo apt install -y espeak-ng tesseract-ocr

echo "=== 安装 Python 依赖 ==="
pip3 install opencv-python pytesseract pillow

echo "=== 验证 ==="
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
python3 -c "import pytesseract; print('pytesseract OK')"
python3 -c "from PIL import Image; print('Pillow OK')"
which espeak-ng && echo "espeak-ng OK"

echo ""
echo "=== 环境就绪，运行 demo ==="
echo "cd task2_sign && python3 task2_main.py --image demo_sign.png"
