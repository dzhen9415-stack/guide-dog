"""
任务2: 标识牌识别模块
识别数字(0-9)、字母(A-Z)及其颜色
支持 OCR（Tesseract）和模板匹配两种模式
"""

import cv2
import numpy as np
import pytesseract
from dataclasses import dataclass


@dataclass
class SignResult:
    digit: int | None
    digit_color_cn: str
    letter: str | None
    letter_color_cn: str
    success: bool


# ============================================================
# 颜色识别
# ============================================================

COLOR_RANGES = {
    "红色": ([(0, 50, 50), (10, 255, 255)], [(160, 50, 50), (180, 255, 255)]),
    "绿色": ([(40, 50, 50), (80, 255, 255)],),
    "蓝色": ([(95, 50, 50), (130, 255, 255)],),
    "黄色": ([(22, 50, 50), (35, 255, 255)],),
    "黑色": ([(0, 0, 0), (180, 80, 60)],),
    "白色": ([(0, 0, 200), (180, 30, 255)],),
}


def classify_color(hsv_frame: np.ndarray, mask: np.ndarray = None) -> str:
    """判断 mask 区域内像素主色"""
    if mask is not None and mask.sum() > 0:
        pixels = hsv_frame[mask > 0]
    else:
        pixels = hsv_frame.reshape(-1, 3)

    if len(pixels) == 0:
        return "未知"

    h, s, v = pixels.mean(axis=0)

    for name, ranges_list in COLOR_RANGES.items():
        for (lo, hi) in ranges_list:
            if lo[0] <= h <= hi[0] and lo[1] <= s <= hi[1] and lo[2] <= v <= hi[2]:
                return name
    return "未知"


# ============================================================
# 预处理
# ============================================================

def _preprocess_for_ocr(roi_bgr: np.ndarray) -> np.ndarray:
    """预处理 ROI 以提升 OCR 识别率：灰度 → 反转 → OTSU → 膨胀"""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    # 检查背景是亮还是暗
    if gray.mean() > 127:
        inv = cv2.bitwise_not(gray)
    else:
        inv = gray
    _, thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(thresh, kernel, iterations=1)


# ============================================================
# 识别方法1: Tesseract OCR（比赛用——印刷体标识牌）
# ============================================================

def ocr_digit_tesseract(roi_bgr: np.ndarray) -> int | None:
    pre = _preprocess_for_ocr(roi_bgr)
    # 尝试多种 PSM 和字符白名单组合
    for psm in [6, 7, 10, 8, 13]:
        # 先尝试纯数字白名单
        text = pytesseract.image_to_string(
            pre, config=f'--psm {psm} -c tessedit_char_whitelist=0123456789'
        ).strip()
        if text.isdigit():
            return int(text)
        # 无白名单，全字符识别后过滤
        text = pytesseract.image_to_string(pre, config=f'--psm {psm}').strip()
        # 去掉非数字字符
        digits_only = ''.join(c for c in text if c.isdigit())
        if digits_only:
            return int(digits_only[-1] if len(digits_only) > 1 else digits_only)
    return None


def ocr_letter_tesseract(roi_bgr: np.ndarray) -> str | None:
    pre = _preprocess_for_ocr(roi_bgr)
    for psm in [6, 7, 10, 8, 13]:
        # 纯字母白名单
        text = pytesseract.image_to_string(
            pre, config=f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        ).strip().upper()
        # 取第一个字母
        for c in text:
            if c.isalpha():
                return c
        # 无白名单
        text = pytesseract.image_to_string(pre, config=f'--psm {psm}').strip().upper()
        for c in text:
            if c.isalpha():
                return c
    return None


# ============================================================
# 识别方法2: 模板匹配（本地测试用——匹配 OpenCV 渲染字体）
# ============================================================

def _make_templates(font_scale=5.5, thickness=10):
    """预生成 0-9 和 A-Z 的模板图像"""
    templates = {}
    size = 200
    for char in list("0123456789") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        (w, h), _ = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = (size - w) // 2
        y = (size + h) // 2
        cv2.putText(img, char, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(cv2.bitwise_not(gray))
        if coords is not None:
            x2, y2, w2, h2 = cv2.boundingRect(coords)
            gray = gray[y2:y2 + h2, x2:x2 + w2]
        templates[char] = cv2.resize(gray, (80, 100))
    return templates


def _get_templates():
    return _make_templates()


def ocr_template(roi_bgr: np.ndarray) -> dict:
    """
    用模板匹配识别单个字符。
    返回 {"digit": int|None, "letter": str|None, "best_char": str, "conf": float}
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() > 127:
        inv = cv2.bitwise_not(gray)
    else:
        inv = gray
    # 裁剪有效区域
    coords = cv2.findNonZero(inv)
    if coords is None:
        return {"digit": None, "letter": None, "best_char": "?", "conf": 0}
    x, y, w, h = cv2.boundingRect(coords)
    roi_cropped = inv[y:y + h, x:x + w]
    roi_resized = cv2.resize(roi_cropped, (80, 100))

    templates = _get_templates()
    best_char = "?"
    best_score = -1

    for char, tmpl in templates.items():
        if tmpl.shape[0] > roi_resized.shape[0] or tmpl.shape[1] > roi_resized.shape[1]:
            continue
        result = cv2.matchTemplate(roi_resized, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val
            best_char = char

    digit = int(best_char) if best_char.isdigit() else None
    letter = best_char if best_char.isalpha() else None
    return {"digit": digit, "letter": letter, "best_char": best_char, "conf": best_score}


# ============================================================
# 标识牌检测
# ============================================================

def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_signboard_roi(frame: np.ndarray):
    """
    检测标识牌并返回透视校正后的 ROI。
    返回 ROI 或 None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_area = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > 1000 and area > best_area:
            best = approx
            best_area = area

    if best is None:
        return None

    pts = best.reshape(4, 2)
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    max_w = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    max_h = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

    dst = np.array([
        [0, 0], [max_w - 1, 0],
        [max_w - 1, max_h - 1], [0, max_h - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect.astype("float32"), dst)
    return cv2.warpPerspective(frame, M, (max_w, max_h))


def split_char_regions(sign_roi: np.ndarray):
    """
    按左右分区提取两个字符区域。
    实际标识牌上数字在左、字母在右，这里直接按水平中线分割。
    """
    h, w = sign_roi.shape[:2]
    mid = w // 2
    # 左右各留一些边距
    margin = int(w * 0.05)
    left = sign_roi[:, margin:mid - margin]
    right = sign_roi[:, mid + margin:w - margin]
    return left, right


# ============================================================
# 主识别流程
# ============================================================

def recognize(frame: np.ndarray, use_tesseract: bool = False) -> SignResult:
    """
    从一帧图像中识别标识牌。
    use_tesseract=False 时使用模板匹配（测试用）
    use_tesseract=True 时使用 Tesseract OCR（比赛用）
    """
    # Step 1: 找标识牌ROI（找不到就用全图）
    sign_roi = find_signboard_roi(frame)
    search_frame = sign_roi if sign_roi is not None else frame

    # Step 2: 左右对半分割
    left_roi, right_roi = split_char_regions(search_frame)
    # 给字符区域加白边（提升OCR率）
    left_roi = cv2.copyMakeBorder(left_roi, 20, 20, 20, 20,
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
    right_roi = cv2.copyMakeBorder(right_roi, 20, 20, 20, 20,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Step 3: 识别字符
    if use_tesseract:
        digit = ocr_digit_tesseract(left_roi)
        letter = ocr_letter_tesseract(right_roi)
    else:
        lr = ocr_template(left_roi)
        rr = ocr_template(right_roi)
        digit = lr["digit"]
        letter = rr["letter"]

    # Step 4: 颜色识别 — 取非背景色像素
    def _extract_fg_color(roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 背景是白色/亮灰色，文字是前景色
        _, fg_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        return classify_color(hsv, fg_mask)

    digit_color = _extract_fg_color(left_roi)
    letter_color = _extract_fg_color(right_roi)

    success = digit is not None and letter is not None
    return SignResult(digit=digit, digit_color_cn=digit_color,
                      letter=letter, letter_color_cn=letter_color, success=success)
