"""World-frame wrist-trajectory rendering (ACE-Ego-Hand Fig. S3 style) for an RLHND checkpoint.

Predicted camera-frame meshes are lifted to the world frame with the dataset's ground-truth
camera poses (cTw). The wrist trajectory is drawn as a fading tube per hand, and the full
hand mesh is rendered at the last frame only; left hand blue, right hand pink.
Camera framing and checkerboard are derived from the GT meshes so the view is identical
across models on the same clip.

python scripts/scripts_eval/viz_traj3d.py --dataset ARCTICEGO-CLIP-VAL --clip_idx 2 \
    --model_path logs/<exp>/checkpoints/stage1_20k.ckpt --out tmp/viz/traj3d_arctic2.png
"""
import argparse
import os

import numpy as np
import pyrootutils
import torch

root = pyrootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True, dotenv=True)
os.chdir(root)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import pyrender  # noqa: E402
import trimesh  # noqa: E402

from models_clip.configs import dataset_config  # noqa: E402
from models_clip.datasets import collate_clips, create_dataset  # noqa: E402
from scripts.scripts_eval.viz_compare import baseline_preds, load_model, predict_ours  # noqa: E402


def gather_cTw(ds, idx, frames):
    """Per-frame world->camera 4x4 for window `idx`, from the raw label chunks."""
    si, _ = ds.windows[idx] if ds.windows is not None else (idx, None)
    seq = ds.sequences[si]
    N = len(frames)
    cTw = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    got = np.zeros(N, bool)
    for hand in ("0", "1"):
        for rel, st, en in seq["hands"].get(hand, []):
            sel = np.where((frames >= st) & (frames < en) & ~got)[0]
            if not len(sel):
                continue
            lab = ds._load_label(rel)
            if "cTw" not in lab:
                continue
            fi = lab["frame_idx"]
            pos = np.clip(np.searchsorted(fi, frames[sel]), 0, len(fi) - 1)
            ok = fi[pos] == frames[sel]
            cTw[sel[ok]] = lab["cTw"][pos[ok]].astype(np.float32)
            got[sel[ok]] = True
    return cTw, got


def gather_kp3d(ds, idx, frames):
    """Per-frame camera-frame GT 3D keypoints (N,2,21,3) + validity, from the raw label chunks."""
    si, _ = ds.windows[idx] if ds.windows is not None else (idx, None)
    seq = ds.sequences[si]
    N = len(frames)
    J = np.zeros((N, 2, 21, 3), np.float32)
    ok_out = np.zeros((N, 2), bool)
    for hand in ("0", "1"):
        s = int(hand)
        got = np.zeros(N, bool)
        for rel, st, en in seq["hands"].get(hand, []):
            sel = np.where((frames >= st) & (frames < en) & ~got)[0]
            if not len(sel):
                continue
            lab = ds._load_label(rel)
            if "hand_keypoints_3d" not in lab:
                continue
            fi = lab["frame_idx"]
            pos = np.clip(np.searchsorted(fi, frames[sel]), 0, len(fi) - 1)
            ok = fi[pos] == frames[sel]
            J[sel[ok], s] = np.asarray(lab["hand_keypoints_3d"])[pos[ok], :21, :3].astype(np.float32)
            got[sel[ok]] = True
        # all-zero keypoints = unlabeled frame
        ok_out[:, s] = got & (np.abs(J[:, s]).max((1, 2)) > 1e-6)
    return J, ok_out


SKEL_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def skeleton_mesh(J, base, r_joint, r_bone):
    """Joint spheres + bone cylinders for one 21-joint hand, flat-colored like the mesh base color."""
    rgba = np.array([*(np.asarray(base) * 255), 255], np.uint8)
    parts = []
    for p in J:
        s = trimesh.creation.icosphere(subdivisions=1, radius=r_joint)
        s.apply_translation(p)
        s.visual.vertex_colors = np.tile(rgba, (len(s.vertices), 1))
        parts.append(s)
    for a, b in SKEL_BONES:
        if np.linalg.norm(J[b] - J[a]) > 1e-6:
            c = trimesh.creation.cylinder(radius=r_bone, segment=[J[a], J[b]], sections=12)
            c.visual.vertex_colors = np.tile(rgba, (len(c.vertices), 1))
            parts.append(c)
    return trimesh.util.concatenate(parts)


def checkerboard(center, up_axis, size=1.6, n=12, z=0.0):
    """Checkerboard plane meshes (two colors) lying on the plane up_axis = z."""
    tiles_a, tiles_b = [], []
    s = size / n
    ax = [i for i in range(3) if i != up_axis]
    for i in range(n):
        for j in range(n):
            v = np.zeros((4, 3))
            v[:, ax[0]] = center[ax[0]] - size / 2 + np.array([i, i + 1, i + 1, i]) * s
            v[:, ax[1]] = center[ax[1]] - size / 2 + np.array([j, j, j + 1, j + 1]) * s
            v[:, up_axis] = z
            m = trimesh.Trimesh(v, [[0, 1, 2], [0, 2, 3]], process=False)
            (tiles_a if (i + j) % 2 == 0 else tiles_b).append(m)
    return trimesh.util.concatenate(tiles_a), trimesh.util.concatenate(tiles_b)


def traj_tube_parts(track, base, r_seg=0.0035, r_pt=0.005, max_gap=3):
    """Per-point [sphere(+cylinder)] parts with colors from the FULL track length (stable while growing)."""
    dark = 0.45 * base
    parts = []
    n = len(track)
    for k, (t, p) in enumerate(track):
        w = (k / max(n - 1, 1)) ** 0.6
        col = ((1 - w) * dark + w * base)
        rgba = np.array([*(col * 255), 255], np.uint8)
        seg = []
        sph = trimesh.creation.icosphere(subdivisions=1, radius=r_pt)
        sph.apply_translation(p)
        sph.visual.vertex_colors = np.tile(rgba, (len(sph.vertices), 1))
        seg.append(sph)
        if k + 1 < n:
            t2, p2 = track[k + 1]
            if t2 - t <= max_gap and np.linalg.norm(p2 - p) > 1e-6:
                cyl = trimesh.creation.cylinder(radius=r_seg, segment=[p, p2], sections=10)
                cyl.visual.vertex_colors = np.tile(rgba, (len(cyl.vertices), 1))
                seg.append(cyl)
        parts.append(trimesh.util.concatenate(seg))
    return parts


def traj_tube(track, base, r_seg=0.0035, r_pt=0.005, max_gap=3):
    """One trimesh tube through `track` [(frame, point)] with colors fading toward `base` over time.

    Consecutive points more than `max_gap` frames apart are not connected (detector gaps)."""
    # fade dark->bright within the SAME hue (never blend toward white: that reads pink at small scale)
    dark = 0.45 * base
    parts = []
    n = len(track)
    for k, (t, p) in enumerate(track):
        w = (k / max(n - 1, 1)) ** 0.6
        col = ((1 - w) * dark + w * base)
        rgba = np.array([*(col * 255), 255], np.uint8)
        sph = trimesh.creation.icosphere(subdivisions=1, radius=r_pt)
        sph.apply_translation(p)
        sph.visual.vertex_colors = np.tile(rgba, (len(sph.vertices), 1))
        parts.append(sph)
        if k + 1 < n:
            t2, p2 = track[k + 1]
            if t2 - t <= max_gap and np.linalg.norm(p2 - p) > 1e-6:
                cyl = trimesh.creation.cylinder(radius=r_seg, segment=[p, p2], sections=10)
                cyl.visual.vertex_colors = np.tile(rgba, (len(cyl.vertices), 1))
                parts.append(cyl)
    return trimesh.util.concatenate(parts)


@torch.no_grad()
def main(args):
    device = torch.device("cuda")
    model, cfg = load_model(args.model_path, device)
    dcfg = dataset_config(cfg.get("DATASETS_CONFIG", "datasets_clip.yaml"))
    ds = create_dataset(cfg, dcfg, args.dataset, train=False)
    b = collate_clips([ds[args.clip_idx]])
    bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
    N = int(b["frame_valid"].sum())
    faces_override = {}
    if args.gt_skel:
        Jk, ok_k = gather_kp3d(ds, args.clip_idx, np.arange(int(b["t0"][0]) if "t0" in b else 0,
                                                            (int(b["t0"][0]) if "t0" in b else 0) + N))
        pred = dict(V=Jk, exist=ok_k, V_gt=np.zeros((N, 2, 778, 3), np.float16))
        print(f"GT skeleton: L {int(ok_k[:, 0].sum())} / R {int(ok_k[:, 1].sum())} labeled frames")
    elif args.gt:
        fcache = os.path.join("tmp/viz_cache", f"{args.frame_exp}__{args.dataset}__{args.clip_idx:05d}.npz")
        Vg = np.load(fcache)["V_gt"].astype(np.float32)[:N]
        exist_g = (b["has_mano"][0].numpy() > 0)[:N]
        # drop junk zero-kp3d placeholder meshes parked near the camera origin
        ok = np.linalg.norm(Vg.mean(2), axis=-1) > 0.2
        pred = dict(V=Vg, exist=exist_g & ok, V_gt=Vg)
    elif args.handflow_viz:
        hd = np.load(args.handflow_viz)
        rows = np.where(hd["clip_idx"] == args.clip_idx)[0]
        assert len(rows), f"clip {args.clip_idx} not in {args.handflow_viz}"
        r = int(rows[0])
        lo = int(hd["win_len"][:r].sum()); hi = lo + int(hd["win_len"][r])
        pred = dict(V=hd["verts_cam"][lo:hi][:N].astype(np.float32),
                    exist=hd["valid"][lo:hi][:N].astype(bool),
                    V_gt=np.zeros((N, 2, 778, 3), np.float16))
        hfF = hd["faces"].astype(np.int64)
        faces_override = {0: hfF[:, ::-1].copy(), 1: hfF}  # mirrored left slot flips winding
    elif args.handflow:
        hd = np.load(args.handflow)
        side = 1 if ("side" not in hd.files or str(hd["side"]) != "left") else 0
        Vh = np.zeros((N, 2, 778, 3), np.float32)
        exist_h = np.zeros((N, 2), bool)
        n = min(N, len(hd["verts_cam"]))
        Vh[:n, side] = hd["verts_cam"][:n]
        exist_h[:n, side] = hd["pred_valid"][:n].astype(bool)
        pred = dict(V=Vh, exist=exist_h, V_gt=np.zeros((N, 2, 778, 3), np.float16))
        faces_override[side] = hd["faces"].astype(np.int64)
    elif args.baseline:
        imgnames = list(b["imgname"][0])
        pred = baseline_preds(args.baseline, imgnames, N, model.mano, device)
    else:
        exp = args.exp or os.path.basename(os.path.dirname(os.path.dirname(args.model_path)))
        cache = os.path.join("tmp/viz_cache", f"{exp}__{args.dataset}__{args.clip_idx:05d}.npz")
        pred = predict_ours(model, bb, N, cache)
    # GT for view framing always comes from one fixed cache so every model shares the camera
    frame_cache = os.path.join("tmp/viz_cache", f"{args.frame_exp}__{args.dataset}__{args.clip_idx:05d}.npz")
    V_gt = np.load(frame_cache)["V_gt"] if os.path.exists(frame_cache) else pred["V_gt"]
    faces_lr = {0: model.mano.faces_left.cpu().numpy(), 1: model.mano.faces.cpu().numpy()}
    Jreg = getattr(model.mano, "J_regressor", None)
    Jreg = Jreg.detach().cpu().numpy() if Jreg is not None else None

    t0 = int(b["t0"][0]) if "t0" in b else 0
    frames = np.arange(t0, t0 + N)
    cTw, got = gather_cTw(ds, args.clip_idx, frames)
    print(f"cTw available on {got.sum()}/{N} frames")

    # lift predicted meshes to world; keep per-hand wrist track + per-frame world meshes
    wrists = {0: [], 1: []}
    last = {0: None, 1: None}
    world_V = {0: {}, 1: {}}
    for t in range(N):
        if not got[t]:
            continue
        wTc = np.linalg.inv(cTw[t])
        for s in range(2):
            if not pred["exist"][t, s]:
                continue
            V = pred["V"][t, s].astype(np.float64)
            Vw = V @ wTc[:3, :3].T + wTc[:3, 3]
            if args.gt_skel:
                wr = Vw[0]  # joint 0 = wrist
            else:
                wr = (Jreg[0] @ Vw) if Jreg is not None else Vw.mean(0)
            wrists[s].append((t, wr))
            last[s] = Vw
            world_V[s][t] = Vw

    assert any(len(wrists[s]) for s in (0, 1)), "no predicted meshes"

    # model-independent framing: GT meshes lifted to world define center/scale/floor.
    # Only hands with actual GT keypoints count: V_gt holds junk zero-pose meshes for
    # unlabeled hands (kp3d all zeros), which otherwise blow up the framing.
    gt_pts = []
    for t in range(N):
        if not got[t]:
            continue
        wTc = np.linalg.inv(cTw[t])
        for s in range(2):
            Vg = V_gt[t, s].astype(np.float64)
            # a hand whose centroid sits within 20cm of the camera is a junk zero-kp3d
            # placeholder (missing GT), not a real hand - drop it from the framing
            if np.abs(Vg).max() < 1e-6 or np.linalg.norm(Vg.mean(0)) < 0.2:
                continue
            gt_pts.append(Vg @ wTc[:3, :3].T + wTc[:3, 3])
    assert gt_pts, "no GT meshes to frame the view"
    allv = np.concatenate(gt_pts)
    print(f"GT framing: {len(gt_pts)} hand-frames, world extent {np.ptp(allv, 0).round(3)}")
    up = args.up_axis
    pred_pts = np.concatenate([np.asarray(p)[None] for s in (0, 1) for _, p in wrists[s]])
    print(f"PRED_WORLD_MIN {min(pred_pts[:, up].min(), allv[:, up].min()):.4f}")
    floor_z = (args.floor_y if args.floor_y is not None else allv[:, up].min()) - 0.02
    ctr = allv.mean(0)

    scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.45] * 3)
    fa, fb = checkerboard(ctr, up, size=max(1.2, 2.2 * np.ptp(allv[:, [i for i in range(3) if i != up]]).max()), z=floor_z)
    for tiles, col in ((fa, (0.92, 0.92, 0.94, 1.0)), (fb, (0.80, 0.80, 0.84, 1.0))):
        scene.add(pyrender.Mesh.from_trimesh(tiles, material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=col, roughnessFactor=1.0, doubleSided=True), smooth=False))
    base = {0: np.array([0.23, 0.42, 0.80]), 1: np.array([0.91, 0.45, 0.62])}   # L blue, R pink (last-frame mesh)
    traj_base = {0: np.array([0.00, 0.30, 1.00]), 1: np.array([1.00, 0.05, 0.05])}  # vivid tube colors
    for s in (0, 1):
        if not wrists[s]:
            continue
        scene_r = max(np.linalg.norm(allv - ctr, axis=1).max(), 0.3)
        tube = traj_tube([(t, np.asarray(p)) for t, p in wrists[s]], traj_base[s],
                         r_seg=max(0.0035, 0.010 * scene_r), r_pt=max(0.005, 0.014 * scene_r))
        scene.add(pyrender.Mesh.from_trimesh(tube, smooth=False))
        if args.gt_skel:
            m = skeleton_mesh(last[s], base[s], r_joint=max(0.004, 0.011 * scene_r), r_bone=max(0.0025, 0.007 * scene_r))
            scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
        else:
            m = trimesh.Trimesh(last[s], faces_override.get(s, faces_lr[s]), process=False)
            scene.add(pyrender.Mesh.from_trimesh(m, material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=(*base[s], 1.0), metallicFactor=0.05, roughnessFactor=0.7), smooth=True))

    # oblique orbit camera around the GT trajectory (identical for every model on this clip)
    rad = max(np.linalg.norm(allv - ctr, axis=1).max(), 0.3) * args.rad_scale
    aa, ee = np.deg2rad(args.az), np.deg2rad(args.el)
    horiz = [i for i in range(3) if i != up]
    eye = ctr.copy()
    eye[horiz[0]] += rad * np.cos(aa) * np.cos(ee)
    eye[horiz[1]] += rad * np.sin(aa) * np.cos(ee)
    eye[up] += rad * np.sin(ee)
    fwd = ctr - eye; fwd /= np.linalg.norm(fwd)
    upv = np.zeros(3); upv[up] = 1.0
    x = np.cross(fwd, upv); x /= np.linalg.norm(x)
    y = np.cross(fwd, x)
    pose = np.eye(4)   # pyrender/GL: camera looks down -z, y up
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, -y, -fwd, eye
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(40))
    scene.add(cam, pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
    r = pyrender.OffscreenRenderer(args.res, int(args.res * 0.62))

    if args.video_out:
        # animated version: the trajectory grows and the hand mesh moves frame by frame,
        # in the SAME world scene (floor, camera, colors) as the static figure.
        import subprocess
        scene_r = max(np.linalg.norm(allv - ctr, axis=1).max(), 0.3)
        tube_parts, track_frames = {}, {}
        for s in (0, 1):
            if not wrists[s]:
                continue
            track = [(t, np.asarray(p)) for t, p in wrists[s]]
            tube_parts[s] = traj_tube_parts(track, traj_base[s],
                                            r_seg=max(0.0035, 0.010 * scene_r), r_pt=max(0.005, 0.014 * scene_r))
            track_frames[s] = [t for t, _ in track]
        frames_out = []
        for t in range(N):
            sc = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.45] * 3)
            for tiles, col in ((fa, (0.92, 0.92, 0.94, 1.0)), (fb, (0.80, 0.80, 0.84, 1.0))):
                sc.add(pyrender.Mesh.from_trimesh(tiles, material=pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=col, roughnessFactor=1.0, doubleSided=True), smooth=False))
            sc.add(cam, pose=pose)
            sc.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
            for s in tube_parts:
                k = sum(1 for f in track_frames[s] if f <= t)
                if k > 0:
                    sc.add(pyrender.Mesh.from_trimesh(trimesh.util.concatenate(tube_parts[s][:k]), smooth=False))
                if t in world_V[s]:
                    Vt = world_V[s][t]
                    if args.gt_skel:
                        m = skeleton_mesh(Vt, base[s], r_joint=max(0.004, 0.011 * scene_r), r_bone=max(0.0025, 0.007 * scene_r))
                        sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
                    else:
                        m = trimesh.Trimesh(Vt, faces_override.get(s, faces_lr[s]), process=False)
                        sc.add(pyrender.Mesh.from_trimesh(m, material=pyrender.MetallicRoughnessMaterial(
                            baseColorFactor=(*base[s], 1.0), metallicFactor=0.05, roughnessFactor=0.7), smooth=True))
            color, _ = r.render(sc, flags=pyrender.RenderFlags.RGBA)
            a = color[..., 3:] / 255.0
            frames_out.append((np.full_like(color[..., :3], 255) * (1 - a) + color[..., :3] * a).astype(np.uint8))
        r.delete()
        os.makedirs(os.path.dirname(args.video_out) or ".", exist_ok=True)
        Hh, Ww = frames_out[0].shape[:2]
        Hh -= Hh % 2; Ww -= Ww % 2
        ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{Ww}x{Hh}", "-r", str(args.fps), "-i", "-",
                               "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", args.video_out], stdin=subprocess.PIPE)
        for fimg in frames_out:
            ff.stdin.write(np.ascontiguousarray(fimg[:Hh, :Ww]).tobytes())
        ff.stdin.close()
        assert ff.wait() == 0, "ffmpeg encode failed"
        print("wrote", args.video_out, f"(clip {b['clipname'][0]}, {N} frames)")
        return

    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    a = color[..., 3:] / 255.0
    color = (np.full_like(color[..., :3], 255) * (1 - a) + color[..., :3] * a).astype(np.uint8)
    r.delete()
    import cv2
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    print("wrote", args.out, f"(clip {b['clipname'][0]}, {sum(len(wrists[s]) for s in (0,1))} wrist pts)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ARCTICEGO-CLIP-VAL")
    p.add_argument("--clip_idx", type=int, default=2)
    p.add_argument("--model_path", required=True)
    p.add_argument("--exp", default=None, help="cache key override: render this experiment's cached predictions")
    p.add_argument("--baseline", default=None, help="baseline dump npz: render these predictions instead of a cache")
    p.add_argument("--handflow", default=None, help="HandFlow pred.npz (verts_cam/pred_valid/faces, single hand)")
    p.add_argument("--gt", action="store_true", help="render the GT trajectory/mesh from the frame_exp cache instead of predictions")
    p.add_argument("--gt_skel", action="store_true", help="GT as 21-joint skeleton + trajectory (datasets without GT MANO, e.g. EgoDex)")
    p.add_argument("--handflow_viz", default=None, help="HandFlow viz-windows npz (verts_cam (F,2,778,3)/valid/win_len/clip_idx/faces)")
    p.add_argument("--frame_exp", default="realhand_v2_cosmos3_nano_hot3dfix_h100_260901",
                   help="experiment whose cached V_gt frames the camera (shared across models)")
    p.add_argument("--up_axis", type=int, default=2)
    p.add_argument("--az", type=float, default=35)
    p.add_argument("--el", type=float, default=22)
    p.add_argument("--rad_scale", type=float, default=2.6, help="camera distance multiple of the GT extent")
    p.add_argument("--floor_y", type=float, default=None, help="fixed floor height along up axis (world); overrides the GT-min default")
    p.add_argument("--res", type=int, default=1400)
    p.add_argument("--out", default="tmp/viz/traj3d.png")
    p.add_argument("--video_out", default=None, help="write an animated mp4 (growing trajectory + moving hand) instead of a PNG")
    p.add_argument("--fps", type=int, default=15)
    main(p.parse_args())
