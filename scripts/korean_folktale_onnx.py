"""
Korean folktale generation using the Python ONNX reference pipeline.

Purpose: Compare audio quality of the Python ONNX pipeline against the Rust
ONNX pipeline to determine whether quality differences come from fp16
precision (same in both) or from the Rust implementation itself.

Usage:
    python scripts/korean_folktale_onnx.py
    python scripts/korean_folktale_onnx.py --speaker kr-spk1_man
"""

import argparse
import os
import sys
import time

# Add the ONNX model dir to path so we can import from vibevoice_full_onnx
MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nenad1002--microsoft-vibevoice-0.5B-onnx-fp16"
    "/snapshots/1227c4cbf8a3f034824ee77d479a13fdd88c17cd"
)
sys.path.insert(0, MODEL_DIR)

import numpy as np
import soundfile as sf
from vibevoice_full_onnx import (
    generate,
    load_config,
    load_schedule,
    load_sessions,
    load_voice,
)
from tokenizers import Tokenizer


# ---------------------------------------------------------------------------
# Story script -- 해와 달이 된 오누이 (The Sun and the Moon)
# Same 20 segments as korean_folktale.rs
# ---------------------------------------------------------------------------

PAUSE_BETWEEN = 1.0  # seconds
PAUSE_SECTION = 1.8  # seconds

SEGMENTS = [
    # -- 도입 (Opening) --
    {
        "text": (
            "옛날 옛적에, 깊은 산골 마을에 어머니와 오누이가 살고 있었습니다. "
            "아버지는 일찍 돌아가시고, 어머니는 먼 마을 잔치집에서 일을 하며 "
            "두 남매를 키우고 있었습니다."
        ),
        "label": "도입: 산골 마을의 가족",
        "pause_after": PAUSE_SECTION,
    },
    # -- 어머니의 귀갓길 (Mother's journey home) --
    {
        "text": (
            "어느 날, 어머니는 잔치집에서 일을 마치고 떡을 한 함지박 "
            "이고 집으로 돌아오고 있었습니다. 달빛 아래 산길을 걸어가는데, "
            "갑자기 큰 호랑이 한 마리가 나타났습니다."
        ),
        "label": "어머니의 귀갓길",
        "pause_after": PAUSE_BETWEEN,
    },
    {
        "text": (
            "호랑이가 말했습니다. "
            "떡 하나 주면 안 잡아먹지! "
            "어머니는 무서워서 떡 하나를 주었습니다. "
            "호랑이는 떡을 받아먹고 사라졌습니다."
        ),
        "label": "호랑이의 첫 번째 요구",
        "pause_after": PAUSE_BETWEEN,
    },
    {
        "text": (
            "그런데 다음 고개를 넘으니 호랑이가 또 나타났습니다. "
            "떡 하나 주면 안 잡아먹지! "
            "어머니는 또 떡을 하나 주었습니다. 이렇게 고개를 넘을 때마다 "
            "호랑이가 나타나서 떡을 하나씩 빼앗아 갔습니다."
        ),
        "label": "반복되는 호랑이의 요구",
        "pause_after": PAUSE_BETWEEN,
    },
    {
        "text": (
            "마침내 떡이 모두 떨어졌습니다. "
            "호랑이가 또 나타나 말했습니다. "
            "떡이 없으면 너를 잡아먹겠다! "
            "불쌍한 어머니는 결국 호랑이에게 잡혀 먹히고 말았습니다."
        ),
        "label": "어머니의 최후",
        "pause_after": PAUSE_SECTION,
    },
    # -- 호랑이, 오누이를 찾아오다 (Tiger finds the children) --
    {
        "text": (
            "호랑이는 어머니의 옷을 입고 머리에 수건을 쓰고 "
            "오누이가 사는 집으로 찾아갔습니다. "
            "애들아, 엄마 왔다. 문 열어라! "
            "하고 문을 두드렸습니다."
        ),
        "label": "호랑이의 변장",
        "pause_after": PAUSE_BETWEEN,
    },
    {
        "text": (
            "오빠가 문틈으로 밖을 내다보았습니다. "
            "우리 엄마 손은 이렇게 까칠까칠하지 않아요! "
            "호랑이는 얼른 손에 밀가루를 묻히고 다시 왔습니다. "
            "자, 보렴. 엄마 손이란다."
        ),
        "label": "오누이의 의심",
        "pause_after": PAUSE_BETWEEN,
    },
    {
        "text": (
            "누이동생이 문틈으로 다시 보았습니다. "
            "우리 엄마 목소리는 이렇게 굵지 않아요! "
            "하지만 호랑이는 목소리를 가늘게 바꾸어 "
            "다시 말했습니다. 감기에 걸려서 그래. 어서 문 열어라."
        ),
        "label": "계속되는 의심과 속임수",
        "pause_after": PAUSE_BETWEEN,
    },
    {
        "text": (
            "결국 오누이는 문을 열고 말았습니다. "
            "그 순간 호랑이의 정체가 드러났습니다! "
            "오누이는 깜짝 놀라 뒷문으로 도망쳐 "
            "마당의 큰 나무 위로 올라갔습니다."
        ),
        "label": "호랑이의 정체 발각",
        "pause_after": PAUSE_SECTION,
    },
    # -- 나무 위의 오누이 (Children in the tree) --
    {
        "text": (
            "호랑이가 나무 아래에서 올려다보며 물었습니다. "
            "너희들 어떻게 올라갔니? "
            "오빠가 꾀를 내어 말했습니다. "
            "손에 참기름을 바르고 올라오면 돼요!"
        ),
        "label": "오빠의 꾀",
        "pause_after": PAUSE_BETWEEN,
    },
    {
        "text": (
            "호랑이는 참기름을 손에 바르고 나무를 올라가려 했지만, "
            "미끄러워서 자꾸 떨어졌습니다. "
            "그때 누이동생이 그만 사실을 말해 버렸습니다. "
            "도끼로 찍으면서 올라오면 돼요!"
        ),
        "label": "누이동생의 실수",
        "pause_after": PAUSE_BETWEEN,
    },
    {
        "text": (
            "호랑이는 도끼를 가져와 나무를 찍으며 올라오기 시작했습니다. "
            "쿵! 쿵! 쿵! "
            "나무가 흔들리고, 호랑이는 점점 가까이 올라왔습니다. "
            "오누이는 너무너무 무서웠습니다."
        ),
        "label": "호랑이가 나무를 올라옴",
        "pause_after": PAUSE_SECTION,
    },
    # -- 하늘의 동아줄 (The rope from heaven) --
    {
        "text": (
            "오누이는 하늘을 향해 간절히 빌었습니다. "
            "하느님, 하느님! 저희를 살려 주시려거든 새 동아줄을 내려 주시고, "
            "죽이시려거든 썩은 동아줄을 내려 주세요!"
        ),
        "label": "하늘에 비는 오누이",
        "pause_after": PAUSE_BETWEEN,
    },
    {
        "text": (
            "그러자 하늘에서 튼튼하고 새하얀 동아줄이 내려왔습니다! "
            "오누이는 동아줄을 꼭 잡고 하늘로 올라갔습니다. "
            "점점 높이, 더 높이 올라가서 구름 위까지 올라갔습니다."
        ),
        "label": "새 동아줄이 내려옴",
        "pause_after": PAUSE_BETWEEN,
    },
    {
        "text": (
            "호랑이도 따라서 빌었습니다. "
            "하느님, 나에게도 동아줄을 내려 주세요! "
            "하늘에서 동아줄이 내려왔지만, "
            "그것은 썩은 동아줄이었습니다."
        ),
        "label": "호랑이에게 내려온 썩은 동아줄",
        "pause_after": PAUSE_BETWEEN,
    },
    {
        "text": (
            "호랑이가 썩은 동아줄을 잡고 올라가다가 "
            "뚝! 줄이 끊어져서 수수밭에 떨어져 죽고 말았습니다. "
            "그래서 수수의 줄기가 빨간 것은 "
            "호랑이의 피가 묻었기 때문이라고 합니다."
        ),
        "label": "호랑이의 최후",
        "pause_after": PAUSE_SECTION,
    },
    # -- 결말 (Ending) --
    {
        "text": (
            "하늘나라에 올라간 오누이는 "
            "하느님으로부터 중요한 임무를 받았습니다. "
            "누이는 해가 되어 세상을 환하게 비추고, "
            "오빠는 달이 되어 밤을 밝히게 되었습니다."
        ),
        "label": "해와 달이 된 오누이",
        "pause_after": PAUSE_BETWEEN,
    },
    {
        "text": (
            "그런데 누이가 말했습니다. "
            "저는 밤이 무서워요. 오빠가 밤을 맡아 주세요. "
            "그래서 오빠가 달이 되고, 누이가 해가 되었다고 합니다."
        ),
        "label": "오빠와 누이의 역할 바꿈",
        "pause_after": PAUSE_BETWEEN,
    },
    {
        "text": (
            "해가 된 누이는 사람들이 자꾸 쳐다보는 것이 부끄러워서 "
            "더욱 밝게 빛나기 시작했답니다. "
            "그래서 우리가 해를 쳐다보면 눈이 부신 것이라고 합니다."
        ),
        "label": "해가 눈부신 이유",
        "pause_after": PAUSE_BETWEEN,
    },
    {
        "text": (
            "이렇게 착한 오누이는 하늘에서 해와 달이 되어 "
            "오래오래 세상을 비추며 살았답니다. "
        ),
        "label": "이야기의 끝",
        "pause_after": 0.0,
    },
]


def make_silence(duration_s: float, sr: int) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Korean folktale TTS via Python ONNX pipeline"
    )
    parser.add_argument("--speaker", default="kr-spk0_woman")
    parser.add_argument("--output-dir", default="output/korean_folktale_onnx")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("  Korean Folktale (Python ONNX): 해와 달이 된 오누이")
    print(f"  Speaker: {args.speaker}")
    print("=" * 72)

    # Load model
    print("Loading ONNX sessions...")
    config = load_config(MODEL_DIR)
    schedule = load_schedule(MODEL_DIR)
    sessions = load_sessions(MODEL_DIR, args.provider)
    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    print("Loading voice preset...")
    voice = load_voice(MODEL_DIR, args.speaker)
    sr = config["sample_rate"]

    n = len(SEGMENTS)
    all_audio = []
    total_audio_s = 0.0
    total_gen_s = 0.0

    for i, seg in enumerate(SEGMENTS, 1):
        label = seg["label"]
        text = seg["text"]
        pause = seg["pause_after"]

        print(f"\n[{i:>2}/{n}] {label}")
        t0 = time.time()
        audio = generate(
            sessions, tokenizer, config, schedule, voice, text, args.cfg_scale
        )
        elapsed = time.time() - t0

        if audio is None:
            print(f"  ERROR: No audio generated")
            continue

        dur = len(audio) / sr
        rtf = elapsed / dur if dur > 0 else 0
        total_audio_s += dur
        total_gen_s += elapsed

        # Save individual segment
        label_short = label.split(":")[0].strip() if ":" in label else label
        seg_file = os.path.join(args.output_dir, f"{i:02d}_{label_short}.wav")
        sf.write(seg_file, audio, sr)

        print(
            f"  {dur:.2f}s audio | {elapsed:.2f}s gen | RTF {rtf:.2f}x | {os.path.basename(seg_file)}"
        )

        all_audio.append(audio)
        if pause > 0:
            all_audio.append(make_silence(pause, sr))

    # Concatenate all segments
    if all_audio:
        combined = np.concatenate(all_audio)
        combined_file = os.path.join(args.output_dir, "해와_달이_된_오누이.wav")
        sf.write(combined_file, combined, sr)
        combined_dur = len(combined) / sr

        print()
        print("=" * 72)
        print("  Summary")
        print("=" * 72)
        print("  Story               : 해와 달이 된 오누이")
        print(f"  Speaker             : {args.speaker}")
        print(f"  Segments synthesized: {n}")
        print(f"  Total speech audio  : {total_audio_s:.1f}s")
        print(f"  Sum of gen times    : {total_gen_s:.1f}s")
        avg_rtf = total_gen_s / total_audio_s if total_audio_s > 0 else 0
        print(f"  Average RTF         : {avg_rtf:.2f}x")
        print(f"  Combined length     : {combined_dur:.1f}s (incl. pauses)")
        print(f"  Output file         : {os.path.abspath(combined_file)}")
        print()


if __name__ == "__main__":
    main()
