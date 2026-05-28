# 智能助盲机器导盲犬 - 竞赛代码

## 环境依赖

```bash
# 基础依赖（云环境通常已有）
sudo apt install -y espeak-ng tesseract-ocr python3-pip

# Python 包
pip3 install opencv-python pytesseract pillow
```

## 任务列表

| 任务 | 状态 | 说明 |
|------|------|------|
| 任务2 | 完成 | 标识牌识别（数字+字母+颜色）+ 语音播报 |

## 任务2 使用

```bash
cd task2_sign

# 单张图片测试
python3 task2_main.py --image demo_sign.png

# 生成测试图并跑 demo
python3 test_sign.py && python3 task2_main.py --image demo_sign.png

# 实时摄像头
python3 task2_main.py --camera 0

# RTSP 流（机器狗摄像头）
python3 task2_main.py --rtsp rtsp://192.168.234.1:8554/test
```
