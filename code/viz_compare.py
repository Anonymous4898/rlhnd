"""Side-by-side mesh comparison video: RLHND checkpoint vs a baseline pred cache.

Layout per method row (like dreamhand_vs_hawor_arctic_ego_s05.mp4):
  [ 2D mesh overlay on the video frame | 3D camera-frame oblique view (GT grey, L blue, R orange, frustum) ]

python scripts/scripts_eval/viz_compare.py \
    --dataset ARCTICEGO-CLIP-VAL --clip_idx 0 \
    --model_path logs/<exp>/checkpoints/epoch=5-step=20000.ckpt --model_label "RLHND (Cosmos 3, 20k)" \
    --baseline_npz /path/to/pred_cache_baselines/hawor__arctic-ego-val.npz --baseline_label "HaWoR (shared detector)" \
    --out tmp/viz/<name>.mp4
"""
import argparse
import os

import numpy as np
import pyrootutils
import torch

root = pyrootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True, dotenv=True)
os.chdir(root)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2  # noqa: E402
import pyrender  # noqa: E402
import trimesh  # noqa: E402

from models_clip.configs import dataset_config, get_config  # noqa: E402
from models_clip.datasets import collate_clips, create_dataset  # noqa: E402
from models_clip.dreamhand import DreamHand  # noqa: E402

L_COL, R_COL, GT_COL = (0.10, 0.35, 0.92, 1.0), (0.90, 0.13, 0.13, 1.0), (0.78, 0.78, 0.78, 1.0)


def load_model(ckpt, device):
    from pathlib import Path
    cfg = get_config(str(Path(ckpt).parent.parent / "model_config.yaml"), merge=True)
    cfg.defrost()
    for k in ("COLOR_AUG_RATE", "FLIP_AUG_RATE", "SCALE_AUG_RATE"):
        cfg.DATASETS.CONFIG[k] = 0.0
    cfg.freeze()
    model = DreamHand.load_from_checkpoint(ckpt, strict=False, weights_only=False, cfg=cfg, map_location="cpu")
    return model.to(device).eval(), cfg


def render_scene(meshes, K, W, H):
    """meshes: list of (verts, faces, rgba) in the camera frame -> RGBA overlay via pyrender."""
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.35] * 3)
    for v, f, c in meshes:
        v = v.copy()
        v[:, 1:] *= -1  # OpenCV -> OpenGL camera
        m = trimesh.Trimesh(v, f, process=False)
        mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=c, metallicFactor=0.1, roughnessFactor=0.7)
        scene.add(pyrender.Mesh.from_trimesh(m, material=mat, smooth=True))
    cam = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.01, zfar=10)
    scene.add(cam, pose=np.eye(4))
    light = pyrender.DirectionalLight(intensity=3.0)
    scene.add(light, pose=np.eye(4))
    r = pyrender.OffscreenRenderer(W, H)
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()
    return color, depth


def overlay_2d(frame, meshes, K):
    H, W = frame.shape[:2]
    color, depth = render_scene(meshes, K, W, H)
    a = color[..., 3:] / 255.0
    return (frame * (1 - a) + color[..., :3] * a).astype(np.uint8)


def oblique_view(meshes_cam, W, H, az=35, el=18, dist_pad=1.25):
    """Render camera-frame meshes + the ego-camera frustum from an oblique viewpoint.

    Transforms everything into the oblique camera's view frame (OpenCV convention) and
    reuses the same IntrinsicsCamera path as the 2D overlay - no hand-rolled GL poses.
    """
    allv = np.concatenate([v for v, _, _ in meshes_cam]) if meshes_cam else np.zeros((1, 3))
    ctr = 0.78 * allv.mean(0)                                     # keep the frustum (origin) in frame
    rad = max(np.linalg.norm(allv - ctr, axis=1).max(), 0.25) * dist_pad + 0.2 * np.linalg.norm(ctr)
    aa, ee = np.deg2rad(az), np.deg2rad(el)
    eye = ctr + np.array([np.sin(aa) * np.cos(ee), -np.sin(ee), -np.cos(aa) * np.cos(ee)]) * rad
    fwd = ctr - eye; fwd /= np.linalg.norm(fwd)                    # z_view (cv: +z forward)
    world_up = np.array([0.0, -1.0, 0.0])                          # cv: y is down -> up is -y
    x = np.cross(fwd, world_up); x /= np.linalg.norm(x)            # x_view (image right)
    y = np.cross(fwd, x); y /= np.linalg.norm(y)                   # y_view (image down)
    R_wc = np.stack([x, y, fwd])                                   # world(cv cam frame) -> view rows

    def to_view(v):
        return (v - eye) @ R_wc.T

    # camera frustum of the ego camera (sits at the origin, looks +z)
    fr = 0.055
    pts = np.array([[0, 0, 0], [-fr, -fr, 2 * fr], [fr, -fr, 2 * fr], [fr, fr, 2 * fr], [-fr, fr, 2 * fr]], float)
    seg = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    fru = trimesh.util.concatenate([trimesh.creation.cylinder(0.004, segment=np.stack([pts[a_], pts[b_]]))
                                    for a_, b_ in seg])
    meshes_view = [(to_view(v), f, c) for v, f, c in meshes_cam]
    meshes_view.append((to_view(fru.vertices), fru.faces, (0.45, 0.45, 0.45, 1.0)))
    K_syn = np.array([[0.95 * H, 0, W / 2], [0, 0.95 * H, H / 2], [0, 0, 1]], float)
    color, _ = render_scene(meshes_view, K_syn, W, H)
    bg = np.full((H, W, 3), 247, np.uint8)
    a3 = color[..., 3:] / 255.0
    return (bg * (1 - a3) + color[..., :3] * a3).astype(np.uint8)


def label_bar(img, text, sub=""):
    H = 30
    bar = np.zeros((H, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        (tw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(bar, sub, (img.shape[1] - tw - 10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


@torch.no_grad()
def predict_ours(model, bb, N, cache_path):
    """Camera-frame pred/GT meshes for one clip; cached to npz for instant re-renders."""
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return {k: d[k] for k in d.files}
    out = model(bb)
    from models_clip.dreamhand.geometry import axis_angle_to_rotmat
    V = (out["V_can"][0, :N] + out["tau"][0, :N][..., None, :]).float().cpu().numpy()
    exist = (torch.sigmoid(out["exist_logit"][0, :N]).cpu().numpy() > 0.5)
    gt_R = axis_angle_to_rotmat(bb["mano_pose"][0:1].float().reshape(1, -1, 2, 16, 3))
    J_gt, V_gt = model.mano(gt_R, bb["betas"][0:1].float(), return_vertices=True)
    tau_gt = bb["kp3d"][0:1, ..., 0, :] - J_gt[..., 0, :]
    V_gt = (V_gt + tau_gt[..., None, :])[0, :N].float().cpu().numpy()
    res = dict(V=V.astype(np.float16), exist=exist, V_gt=V_gt.astype(np.float16))
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, **res)
    return res


def baseline_preds(npz_path, imgnames, N, mano, device):
    """Reconstruct camera-frame meshes for one clip from a baseline dump npz (rot16+betas+J_cam)."""
    import torch as _t
    d = np.load(npz_path)
    # materialize once: NpzFile re-reads (and decompresses) the whole array on EVERY d[key] access,
    # which turns the index build into hours on large merged dumps
    imgn, right = d["imgname"], d["right"]
    rot16_a, betas_a, J_cam_a = d["rot16"], d["betas"], d["J_cam"]
    idx = {(str(imgn[i]), int(right[i])): i for i in range(len(imgn))}
    V = np.zeros((N, 2, 778, 3), np.float32)
    exist = np.zeros((N, 2), bool)
    for t in range(N):
        for sl in range(2):
            j = idx.get((str(imgnames[t]), sl), -1)
            if j < 0:
                continue
            rot = _t.from_numpy(rot16_a[j]).float()
            R_full = _t.eye(3).expand(1, 1, 2, 16, 3, 3).clone()
            R_full[0, 0, sl] = rot
            beta = _t.zeros(1, 2, 10); beta[0, sl] = _t.from_numpy(betas_a[j]).float()
            Jb, Vb = mano(R_full.to(device), beta.to(device), return_vertices=True)
            off = J_cam_a[j][0] - Jb[0, 0, sl, 0].float().cpu().numpy()
            V[t, sl] = Vb[0, 0, sl].float().cpu().numpy() + off
            exist[t, sl] = True
    return dict(V=V.astype(np.float16), exist=exist, V_gt=np.zeros((N, 2, 778, 3), np.float16))


def write_mp4(frames_out, out_path, fps):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Hh, Ww = frames_out[0].shape[:2]
    Hh -= Hh % 2; Ww -= Ww % 2
    import subprocess
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{Ww}x{Hh}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                           "-crf", "18", "-pix_fmt", "yuv420p", out_path], stdin=subprocess.PIPE)
    for f in frames_out:
        ff.stdin.write(np.ascontiguousarray(f[:Hh, :Ww]).tobytes())
    ff.stdin.close()
    assert ff.wait() == 0, "ffmpeg encode failed"
    print("wrote", out_path)


def render_raw_overlays(b, N, preds, labels, faces_lr, K, out_base, fps, skip_existing=False):
    """One plain video per method: the original frames with only the 2D mesh overlay."""
    video = b["video"][0].numpy()
    for pred, lab in zip(preds, labels):
        slug = "".join(c if c.isalnum() else "_" for c in lab.split("(")[0].strip().lower()).strip("_")
        if skip_existing and os.path.exists(f"{out_base}__{slug}.mp4"):
            continue
        flr = pred.get("faces_lr", faces_lr)
        frames_out = []
        for t in range(N):
            frame = video[t].transpose(1, 2, 0).copy()
            meshes = [(pred["V"][t, sl].astype(np.float64), flr[sl], L_COL if sl == 0 else R_COL)
                      for sl in range(2) if pred["exist"][t, sl]]
            frames_out.append(overlay_2d(frame, meshes, K[t]))
        write_mp4(frames_out, f"{out_base}__{slug}.mp4", fps)


def render_clip(b, N, preds, labels, faces_lr, K, out_path, fps, header, overlay_only=False):
    video = b["video"][0].numpy()
    has_mano = b["has_mano"][0].numpy() > 0
    frames_out = []
    for t in range(N):
        frame = video[t].transpose(1, 2, 0).copy()
        rows = []
        for pred, lab in zip(preds, labels):
            meshes = []
            flr = pred.get("faces_lr", faces_lr)
            for sl in range(2):
                if pred["exist"][t, sl]:
                    meshes.append((pred["V"][t, sl].astype(np.float64), flr[sl], L_COL if sl == 0 else R_COL))
            meshes3d = [(preds[0]["V_gt"][t, sl].astype(np.float64), faces_lr[sl], GT_COL) for sl in range(2) if has_mano[t, sl]]
            meshes3d += meshes
            ov = overlay_2d(frame, meshes, K[t])
            if overlay_only:
                row = label_bar(ov, lab, "2D mesh overlay (L blue, R orange)")
            else:
                o3 = oblique_view(meshes3d, ov.shape[1], ov.shape[0])
                row = label_bar(np.concatenate([ov, o3], 1), lab,
                                "2D mesh overlay | 3D camera-frame view (GT grey, L blue, R orange)")
            rows.append(row)
        panel = np.concatenate(rows, 0)
        head = np.zeros((30, panel.shape[1], 3), np.uint8)
        cv2.putText(head, f"{header}  frame {t + 1}/{N}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 1, cv2.LINE_AA)
        frames_out.append(np.concatenate([head, panel], 0))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Hh, Ww = frames_out[0].shape[:2]
    Hh -= Hh % 2; Ww -= Ww % 2
    import subprocess
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{Ww}x{Hh}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                           "-crf", "20", "-pix_fmt", "yuv420p", out_path], stdin=subprocess.PIPE)
    for f in frames_out:
        ff.stdin.write(np.ascontiguousarray(f[:Hh, :Ww]).tobytes())
    ff.stdin.close()
    assert ff.wait() == 0, "ffmpeg encode failed"
    print("wrote", out_path)


@torch.no_grad()
def main(args):
    device = torch.device("cuda")
    model1, cfg = load_model(args.model_path, device)
    model2 = load_model(args.model2_path, device)[0] if args.model2_path else None
    dcfg = dataset_config(cfg.get("DATASETS_CONFIG", "datasets_clip.yaml"))
    ds = create_dataset(cfg, dcfg, args.dataset, train=False)
    faces = (model1.mano.faces_left.cpu().numpy(), model1.mano.faces.cpu().numpy())
    if ":" in args.clips:
        lo, hi = args.clips.split(":")
        idxs = list(range(int(lo), min(int(hi), len(ds))))
    else:
        idxs = [int(x) for x in args.clips.split(",")]
    exp1 = os.path.basename(os.path.dirname(os.path.dirname(args.model_path)))
    exp2 = os.path.basename(os.path.dirname(os.path.dirname(args.model2_path))) if args.model2_path else None
    for idx in idxs:
        try:
            b = collate_clips([ds[idx]])
        except Exception as e:  # noqa: BLE001
            print(f"[skip] clip {idx}: {type(e).__name__}: {e}")
            continue
        bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        N = int(b["frame_valid"].sum())
        K = b["K"][0].numpy()
        preds, labels = [], []
        for m, exp, lab in ((model1, exp1, args.model_label), (model2, exp2, args.model2_label)):
            if m is None:
                continue
            cp = os.path.join(args.cache_dir, f"{exp}__{args.dataset}__{idx:05d}.npz") if args.cache_dir else None
            preds.append(predict_ours(m, bb, N, cp))
            labels.append(lab)
        if args.baseline_rows:
            imgnames = list(b["imgname"][0])
            for spec in args.baseline_rows.split(";"):
                lab, path = spec.split("=", 1)
                preds.append(baseline_preds(path, imgnames, N, model1.mano, device))
                labels.append(lab)
        if args.handflow:
            lab, path = args.handflow.split("=", 1)
            hd = np.load(path)
            side = 1 if ("side" not in hd.files or str(hd["side"]) != "left") else 0
            Vh = np.zeros((N, 2, 778, 3), np.float32)
            exist_h = np.zeros((N, 2), bool)
            n = min(N, len(hd["verts_cam"]))
            Vh[:n, side] = hd["verts_cam"][:n]
            exist_h[:n, side] = hd["pred_valid"][:n].astype(bool)
            hf_faces = hd["faces"].astype(np.int64)
            preds.append(dict(V=Vh, exist=exist_h, faces_lr=(hf_faces, hf_faces)))
            labels.append(lab)
        if args.handflow_viz:
            hd = np.load(args.handflow_viz)
            rows = np.where(hd["clip_idx"] == idx)[0]
            if len(rows):
                r = int(rows[0])
                lo = int(hd["win_len"][:r].sum()); hi = lo + int(hd["win_len"][r])
                Vh = hd["verts_cam"][lo:hi][:N].astype(np.float32)
                exist_h = hd["valid"][lo:hi][:N].astype(bool)
                hfF = hd["faces"].astype(np.int64)
                # slot 0 was predicted mirrored and reflected back (x -> -x), which flips the winding
                preds.append(dict(V=Vh, exist=exist_h, faces_lr=(hfF[:, ::-1].copy(), hfF)))
                labels.append("HandFlow")
            else:
                print(f"[handflow_viz] clip {idx} not in dump; skipping row")
        if args.gt_row:
            preds.append(dict(V=preds[0]["V_gt"].astype(np.float32), exist=b["has_mano"][0].numpy() > 0))
            labels.append("GT")
        clip_tag = str(b["clipname"][0]).replace("/", "_")
        if args.raw_overlay:
            out_base = os.path.join(args.out_dir, args.dataset, f"{idx:05d}_{clip_tag}")
            render_raw_overlays(b, N, preds, labels, faces, K, out_base, args.fps, skip_existing=args.skip_existing)
            continue
        out_path = os.path.join(args.out_dir, args.dataset, f"{idx:05d}_{clip_tag}.mp4")
        header = f"{args.dataset}  {b['clipname'][0]}  t0={int(b['t0'][0]) if 't0' in b else 0}"
        render_clip(b, N, preds, labels, faces, K, out_path, args.fps, header, overlay_only=args.overlay_only)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ARCTICEGO-CLIP-VAL")
    p.add_argument("--clips", default="0", help="comma list or lo:hi range of clip indices")
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_label", default="RLHND (Cosmos 3 Nano, 20k)")
    p.add_argument("--model2_path", default="")
    p.add_argument("--model2_label", default="DreamHand (Wan2.2, 20k)")
    p.add_argument("--cache_dir", default="tmp/viz_cache")
    p.add_argument("--out_dir", default="tmp/viz")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--baseline_rows", default="", help="Label=npz;Label2=npz2 extra rows from baseline dump caches")
    p.add_argument("--handflow", default="", help="Label=pred.npz HandFlow-style dump (verts_cam/pred_valid/faces, single hand)")
    p.add_argument("--gt_row", action="store_true", help="also render a GT-mesh row/video (labeled hands only)")
    p.add_argument("--handflow_viz", default="", help="HandFlow viz-windows npz (verts_cam (F,2,778,3)/valid/win_len/clip_idx/faces)")
    p.add_argument("--skip_existing", action="store_true", help="raw_overlay: do not re-render per-method videos that already exist")
    p.add_argument("--overlay_only", action="store_true")
    p.add_argument("--raw_overlay", action="store_true", help="one plain overlay-only video per method, no labels/headers/3D")
    main(p.parse_args())
