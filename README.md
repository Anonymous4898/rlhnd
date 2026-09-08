# RLHND anonymous project page

- `index.html` — the page (GitHub Pages ready: relative `videos/` paths, Google Fonts only external dep).
- `videos/` — paired clips: `<clip>_<model>_2d.mp4` (mesh overlay on the input video, from
  `code/viz_compare.py --raw_overlay`) and `<clip>_<model>_3d.mp4` (world-frame animated
  trajectory + moving hands, from `code/viz_traj3d.py --video_out`).
- `code/` — render scripts (canonical copies live in the training repo):
  - `viz_traj3d.py` — world-frame renders; `--video_out x.mp4` = animated mode (same camera/floor as the static figures).
  - `viz_compare.py` — 2D overlay videos (`--raw_overlay`, `--gt_row`, `--handflow_viz`).
  - `traj_video_demo.sh` — the exact commands that produced the videos here.
  - `qual_runner.sh`, `make_concat.py` — full qualitative batch + labeled grid, for more clips.

To deploy: push this folder to an anonymous GitHub repo with Pages enabled (root).
