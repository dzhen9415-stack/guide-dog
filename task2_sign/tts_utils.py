"""
语音播报工具 - 基于 espeak-ng
"""

import subprocess
import threading


def speak(text: str, blocking: bool = False):
    """语音播报中文文本"""
    print(f"[TTS] {text}")

    def _run():
        p1 = subprocess.Popen(
            ["espeak-ng", "-v", "zh", "-s", "150", "--stdout", text],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["aplay", "-q"],
            stdin=p1.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        p1.stdout.close()
        p1.wait()

    if blocking:
        _run()
    else:
        t = threading.Thread(target=_run, daemon=True)
        t.start()


def speak_sign_result(digit, digit_color, letter, letter_color):
    """
    按比赛要求格式播报标识牌信息:
    "标识牌上的数字是X，数字颜色是X色，标识牌上的字母是X，字母颜色是X色"
    """
    msg = (
        f"标识牌上的数字是{digit}，"
        f"数字颜色是{digit_color}，"
        f"标识牌上的字母是{letter}，"
        f"字母颜色是{letter_color}"
    )
    speak(msg, blocking=True)  # 同步播报，确保完整播完
