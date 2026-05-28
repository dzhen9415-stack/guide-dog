"""
任务2: 标识牌识别 - 主程序
用法:
  python3 task2_main.py                    # 用摄像头
  python3 task2_main.py --image test.png   # 用图片
  python3 task2_main.py --camera 0         # 指定摄像头编号
  python3 task2_main.py --rtsp rtsp://...  # RTSP 流（机器狗摄像头）
"""

import cv2
import argparse
import time
from sign_recognizer import recognize
from tts_utils import speak, speak_sign_result


def run_on_image(path: str):
    """单张图片识别"""
    frame = cv2.imread(path)
    if frame is None:
        print(f"无法读取图片: {path}")
        return

    print(f"识别图片: {path}")
    result = recognize(frame, use_tesseract=True)

    if result.success:
        print(f"数字={result.digit} 颜色={result.digit_color_cn}")
        print(f"字母={result.letter} 颜色={result.letter_color_cn}")
        speak_sign_result(result.digit, result.digit_color_cn,
                           result.letter, result.letter_color_cn)
    else:
        print("识别失败")
        speak("识别失败，请重新尝试")


def run_on_camera(source):
    """实时摄像头识别 - 按空格键触发识别，按 ESC 退出"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"无法打开摄像头: {source}")
        return

    print("摄像头已打开。")
    print("  按 空格键  触发识别")
    print("  按 ESC/ Q  退出")
    print()

    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 显示当前帧
        display = frame.copy()
        if last_result:
            text = f"数字:{last_result.digit}({last_result.digit_color_cn}) 字母:{last_result.letter}({last_result.letter_color_cn})"
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        cv2.imshow("Task2 - 标识牌识别 (空格=识别, ESC=退出)", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):  # ESC / Q
            break
        elif key == ord(' '):  # 空格 - 触发识别
            print("正在识别...", end=" ", flush=True)
            start = time.time()
            result = recognize(frame, use_tesseract=True)
            elapsed = time.time() - start
            last_result = result

            print(f"({elapsed:.1f}秒)", end=" ")
            if result.success:
                print(f"数字={result.digit}({result.digit_color_cn}) 字母={result.letter}({result.letter_color_cn})")
                speak_sign_result(result.digit, result.digit_color_cn,
                                   result.letter, result.letter_color_cn)
            else:
                print("识别失败")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="任务2: 标识牌识别")
    parser.add_argument("--image", "-i", help="识别单张图片")
    parser.add_argument("--camera", "-c", type=int, default=None, help="摄像头编号")
    parser.add_argument("--rtsp", "-r", help="RTSP 流地址")
    args = parser.parse_args()

    if args.image:
        run_on_image(args.image)
    elif args.rtsp:
        run_on_camera(args.rtsp + "?tcp")
    else:
        cam_id = args.camera if args.camera is not None else 0
        run_on_camera(cam_id)


if __name__ == "__main__":
    main()
