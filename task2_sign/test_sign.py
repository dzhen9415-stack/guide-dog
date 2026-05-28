"""
测试标识牌识别流程（PIL渲染→Tesseract OCR→OpenCV颜色识别）
用法: python3 test_sign.py
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from sign_recognizer import recognize, classify_color

# PIL 颜色映射
PIL_COLORS = {
    "红色": (255, 0, 0),
    "绿色": (0, 200, 0),
    "蓝色": (0, 0, 255),
    "黄色": (200, 200, 0),
    "黑色": (0, 0, 0),
    "白色": (255, 255, 255),
}


def generate_sign_pil(digit: int, letter: str,
                      digit_color: str, letter_color: str) -> np.ndarray:
    """用 PIL 渲染标识牌，更像真实印刷体"""
    w, h = 600, 350
    img = Image.new("RGB", (w, h), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    font = None
    for fp in [
        "/usr/share/fonts/truetype/tlwg/TlwgTypist-Bold.ttf",
        "/usr/share/fonts/truetype/tlwg/Umpush-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            font = ImageFont.truetype(fp, 140)
            break
        except (IOError, OSError):
            continue
    if font is None:
        font = ImageFont.load_default()

    d_rgb = PIL_COLORS.get(digit_color, (0, 0, 0))
    l_rgb = PIL_COLORS.get(letter_color, (0, 0, 0))

    # 左侧数字框
    draw.rectangle([30, 25, 280, 325], outline=(180, 180, 180), width=3)
    draw.text((80, 110), str(digit), fill=d_rgb, font=font)

    # 右侧字母框
    draw.rectangle([310, 25, 570, 325], outline=(180, 180, 180), width=3)
    draw.text((360, 110), letter, fill=l_rgb, font=font)

    # PIL → OpenCV BGR
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def simple_recognize(frame: np.ndarray):
    """
    简化识别：左右对半切 → 分别 OCR + 颜色。
    专门配测试用。
    """
    from sign_recognizer import ocr_digit_tesseract, ocr_letter_tesseract

    h, w = frame.shape[:2]
    mid = w // 2

    # 左右半
    left = frame[30:h - 30, 30:mid - 20]
    right = frame[30:h - 30, mid + 20:w - 30]

    # 白边
    left = cv2.copyMakeBorder(left, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    right = cv2.copyMakeBorder(right, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    digit = ocr_digit_tesseract(left)
    letter = ocr_letter_tesseract(right)

    # 颜色
    def fg_color(roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        return classify_color(hsv, mask)

    dc = fg_color(left)
    lc = fg_color(right)

    return digit, dc, letter, lc


def test():
    cases = [
        (5, "A", "红色", "蓝色"),
        (9, "Z", "绿色", "黄色"),
        (0, "M", "黑色", "红色"),
        (3, "K", "蓝色", "绿色"),
        (7, "B", "红色", "白色"),
        (2, "X", "黄色", "黑色"),
    ]

    correct = 0
    for digit, letter, dc, lc in cases:
        img = generate_sign_pil(digit, letter, dc, lc)
        rd, rdc, rl, rlc = simple_recognize(img)

        d_ok = rd == digit
        l_ok = rl == letter
        if d_ok and l_ok:
            correct += 1

        print(f"期望:{digit}/{dc}/{letter}/{lc} → "
              f"识别:{rd}/{rdc}/{rl}/{rlc} "
              f"[{'OK' if d_ok and l_ok else 'FAIL'}]")

    print(f"\n准确率: {correct}/{len(cases)}")


if __name__ == "__main__":
    test()
