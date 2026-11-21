"""Step 1: record posture images from the webcam.

Usage:   python src/collect_data.py --person rohit

Keys:    1 = good   2 = forward_head   3 = slouching   4 = leaning
         SPACE = stop recording        q = quit
After you press a class key there is a 3 second countdown, so you can get into the posture.
Images are saved as data/raw/<class>/<person>__<timestamp>.jpg
"""
import argparse
import re
import time
import cv2

from config import RAW_DIR, CLASS_NAMES, CAMERA_INDEX, MIRROR

COUNTDOWN = 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True, help="your name (letters/numbers only)")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between saved images")
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX)
    args = ap.parse_args()

    person = re.sub(r"[^a-z0-9]", "", args.person.lower())
    if not person:
        raise SystemExit("Please give a valid --person name.")

    for c in CLASS_NAMES:
        (RAW_DIR / c).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Try --camera 1")

    counts = {c: len(list((RAW_DIR / c).glob(f"{person}__*"))) for c in CLASS_NAMES}
    current, start_t, last_save = None, 0.0, 0.0
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if MIRROR:
            frame = cv2.flip(frame, 1)
        clean = frame.copy()
        now = time.time()
        status = "Press 1-4 to start recording"

        if current is not None:
            elapsed = now - start_t
            if elapsed < COUNTDOWN:
                status = f"Get ready: {current}  {COUNTDOWN - elapsed:.0f}"
            else:
                status = f"RECORDING: {current}"
                if now - last_save >= args.interval:
                    name = f"{person}__{int(now * 1000)}.jpg"
                    cv2.imwrite(str(RAW_DIR / current / name), clean)
                    counts[current] += 1
                    last_save = now

        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        for i, c in enumerate(CLASS_NAMES):
            cv2.putText(frame, f"{i + 1}. {c}: {counts[c]}", (10, 65 + 28 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("Collect posture data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            current = None
        elif ord("1") <= key <= ord(str(len(CLASS_NAMES))):
            current = CLASS_NAMES[key - ord("1")]
            start_t = now

    cap.release()
    cv2.destroyAllWindows()
    print("Saved images:", counts)


if __name__ == "__main__":
    main()
