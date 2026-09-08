"""Tile per-model raw-overlay videos of one clip into a rough labeled grid mp4.

Usage: make_concat.py <prefix> <out.mp4>
  prefix = e.g. tmp/qualitative/ARCTICEGO-CLIP-VAL/00002  (matches <prefix>_*__<slug>.mp4)
"""
import glob
import subprocess
import sys

import cv2
import numpy as np

ORDER = [("gt", "GT"), ("rlhnd", "Ours"), ("dreamhand", "DreamHand"), ("handflow", "HandFlow"),
         ("hawor", "HaWoR"), ("hamer", "HaMeR"), ("wilor", "WiLoR"), ("haptic", "Haptic")]

prefix, out = sys.argv[1], sys.argv[2]
vids = []
for slug, lab in ORDER:
    fs = sorted(glob.glob(f"{prefix}_*__{slug}.mp4"))
    if fs:
        vids.append((fs[0], lab))
assert vids, f"no per-model videos matching {prefix}_*__*.mp4"
caps = [cv2.VideoCapture(f) for f, _ in vids]
fps = caps[0].get(cv2.CAP_PROP_FPS) or 15
n = len(vids)
cols = 4 if n > 4 else n
rows = (n + cols - 1) // cols
TW, TH = 640, 360
frames = []
while True:
    tiles = []
    for c, (_, lab) in zip(caps, vids):
        ok, fr = c.read()
        if not ok:
            tiles = None
            break
        fr = cv2.resize(fr, (TW, TH))
        cv2.rectangle(fr, (0, 0), (TW, 34), (0, 0, 0), -1)
        cv2.putText(fr, lab, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(fr)
    if tiles is None:
        break
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((TH, TW, 3), np.uint8))
    grid = np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], 1) for r in range(rows)], 0)
    frames.append(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
for c in caps:
    c.release()
assert frames, "no frames decoded"
H, W = frames[0].shape[:2]
ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{W}x{H}", "-r", str(int(round(fps))), "-i", "-",
                       "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
for f in frames:
    ff.stdin.write(np.ascontiguousarray(f).tobytes())
ff.stdin.close()
assert ff.wait() == 0, "ffmpeg encode failed"
print("wrote", out, f"({n} tiles, {len(frames)} frames)")
