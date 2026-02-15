# Embodied_WorldModel2
# ====== 0) common ======
REPO=/home/keito/worldmodel_lecture/kadai/Embodied_WorldModel2_keito
SUB=$REPO

cd "$SUB"
export PYTHONPATH="$REPO/src"

# ---- common hyper ----
SEQ=80
H=79
RS=40

RS=40
HZ=10
DT=0.1

# ---- paths ----
TRAIN="datasets/train"
VAL="datasets/val"
INTRAP="datasets/test_intrap"
EXTRAP="datasets/test_extrap"

CKPT_VP="outputs/ckpt_vp/best.pt"
CKPT_VPF="outputs/ckpt_vpf/best.pt"
PROBE_VP="outputs/probes/probe_vp.pt"
PROBE_VPF="outputs/probes/probe_vpf.pt"

# ====== 1) dataset ======
mkdir -p datasets/{train,val,test_extrap,test_intrap}
mkdir -p outputs/{ckpt_vp,ckpt_vpf,probes}
mkdir -p outputs/{genmap,intervention,probe_eval,logs,lists,final_figs}

PYTHONPATH=src python scripts/collect.py \
  --split train \
  --out-root datasets \
  --episodes-per-condition 200 \
  --seed 1 \
  --policy fixed \
  --push-mode vel \
  --control-hz 10 \
  --T 80 \
  --release-steps 40 \
  --release-lift 0.12 \
  --release-backoff 0.04 \
  --no-force-target \
  --kick-steps 0

# 1-2) val
PYTHONPATH=src python scripts/collect.py \
  --split val \
  --out-root datasets \
  --episodes-per-condition 8 \
  --episode-id-start 200000 \
  --seed 2 \
  --policy fixed \
  --push-mode vel \
  --control-hz 10 \
  --T 80 \
  --release-steps 40 \
  --release-lift 0.12 \
  --release-backoff 0.04 \
  --no-force-target \
  --kick-steps 0

# 1-3) test_extrap
PYTHONPATH=src python scripts/collect.py \
  --split test_extrap \
  --out-root datasets \
  --episodes-per-condition 30 \
  --episode-id-start 300000 \
  --seed 1 \
  --policy fixed \
  --push-mode vel \
  --control-hz 10 \
  --T 80 \
  --release-steps 40 \
  --release-lift 0.12 \
  --release-backoff 0.04 \
  --no-force-target \
  --kick-steps 0

PYTHONPATH=src python scripts/collect.py \
  --split test_intrap \
  --out-root datasets \
  --episodes-per-condition 30 \
  --episode-id-start 200000 \
  --seed 2 \
  --policy fixed \
  --push-mode vel \
  --control-hz 10 \
  --T 80 \
  --release-steps 40 \
  --release-lift 0.12 \
  --release-backoff 0.04 \
  --no-force-target \
  --kick-steps 0

# ====== 2) train world models ======

# VP
PYTHONPATH=$REPO/src python scripts/train_wm.py \
  --dataset-root datasets/train \
  --val-root datasets/val \
  --out-dir outputs/ckpt_vp \
  --seed 0 \
  --sequence-length 64 \
  --frame-skip 1 \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --grad-clip 1.0 \
  --log-every 50 \
  --save-norm-cfg

# VPF
PYTHONPATH=$REPO/src python scripts/train_wm.py \
  --dataset-root datasets/train \
  --val-root datasets/val \
  --use-force \
  --out-dir outputs/ckpt_vpf \
  --seed 0 \
  --sequence-length 64 \
  --frame-skip 1 \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --grad-clip 1.0 \
  --log-every 50 \
  --force-lambda-train 0.3 \
  --force-dropout 0.5 \
  --lambda-force 0.3 \
  --lambda-zero 0.1 \
  --w-contact 1.0 \
  --w-coast 1.0 \
  --force-th 1.0 \
  --force-loss-norm 200 \
  --save-norm-cfg

# ====== 3) train probe ======
PYTHONPATH=$REPO/src python scripts/train_probe.py \
  --dataset-root datasets/train \
  --ckpt "$CKPT_VP" \
  --rep-mode mean \
  --batch-size 32 \
  --val-ratio 0.2 \
  --seed 0 \
  --epochs 200 \
  --lr 1e-2 \
  --out outputs/probes/probe_vp_metrics.txt \
  --save-probe outputs/probes/probe_vp.pt \
  --cache-latents \
  --num-workers 8 --pin-memory --persistent-workers --prefetch-factor 4

PYTHONPATH=$REPO/src python scripts/train_probe.py \
  --dataset-root datasets/train \
  --ckpt "$CKPT_VPF" \
  --use-force \
  --rep-mode mean \
  --batch-size 32 \
  --val-ratio 0.2 \
  --seed 0 \
  --epochs 200 \
  --lr 1e-2 \
  --out outputs/probes/probe_vpf_metrics.txt \
  --save-probe outputs/probes/probe_vpf.pt \
  --cache-latents \
  --num-workers 8 --pin-memory --persistent-workers --prefetch-factor 4

# A-1) 各split全エピソードを回して “traj/metrics を吐く”
# 一般化地図作成

# VP（intrap/extrap）
# VP intrap
OUT="outputs/genmap/vp_intrap"
mkdir -p "$OUT"
for ep in "$INTRAP"/episode_*.npz; do
  PYTHONPATH="$REPO/src" python scripts/run_intervention.py \
    --episode-path "$ep" \
    --horizon "$H" \
    --ckpt "$CKPT_VP" \
    --probe "$PROBE_VP" \
    --axis mass --alpha 0.0 \
    --intervene-mode h0 \
    --release-start "$RS" \
    --outdir "$OUT" \
    --tag "genmap_vp_intrap" \
    --overwrite
done

# VP extrap
OUT="outputs/genmap/vp_extrap"
mkdir -p "$OUT"
for ep in "$EXTRAP"/episode_*.npz; do
  PYTHONPATH="$REPO/src" python scripts/run_intervention.py \
    --episode-path "$ep" \
    --horizon "$H" \
    --ckpt "$CKPT_VP" \
    --probe "$PROBE_VP" \
    --axis mass --alpha 0.0 \
    --intervene-mode h0 \
    --release-start "$RS" \
    --outdir "$OUT" \
    --tag "genmap_vp_extrap" \
    --overwrite
done


# A-1-3) VPF / intrap（forceを使うモデル）
# VPF intrap
OUT="outputs/genmap/vpf_intrap"
mkdir -p "$OUT"
for ep in "$INTRAP"/episode_*.npz; do
  PYTHONPATH="$REPO/src" python scripts/run_intervention.py \
    --use-force \
    --episode-path "$ep" \
    --horizon "$H" \
    --ckpt "$CKPT_VPF" \
    --probe "$PROBE_VPF" \
    --axis mass --alpha 0.0 \
    --intervene-mode h0 \
    --release-start "$RS" \
    --force-input-mode auto \
    --h0-force zero \
    --force-feedback hat \
    --force-hat-scale 0.04 \
    --force-cut-after-release \
    --force-decay-lambda 1.0 \
    --contact-th 0.10 \
    --outdir "$OUT" \
    --tag "genmap_vpf_intrap" \
    --overwrite
done

# VPF extrap
OUT="outputs/genmap/vpf_extrap"
mkdir -p "$OUT"
for ep in "$EXTRAP"/episode_*.npz; do
  PYTHONPATH="$REPO/src" python scripts/run_intervention.py \
    --use-force \
    --episode-path "$ep" \
    --horizon "$H" \
    --ckpt "$CKPT_VPF" \
    --probe "$PROBE_VPF" \
    --axis mass --alpha 0.0 \
    --intervene-mode h0 \
    --release-start "$RS" \
    --force-input-mode auto \
    --h0-force zero \
    --force-feedback hat \
    --force-hat-scale 0.04 \
    --force-cut-after-release \
    --force-decay-lambda 1.0 \
    --contact-th 0.10 \
    --outdir "$OUT" \
    --tag "genmap_vpf_extrap" \
    --overwrite
done


# A-2) split 集計（grid_metrics.csv）
# VP intrap
python tools/genmap_split_agg.py \
  --glob-full "outputs/genmap/vp_intrap/*_traj_full.csv" \
  --glob-open "outputs/genmap/vp_intrap/*_traj_open.csv" \
  --out "outputs/genmap/vp_intrap/grid_metrics.csv" \
  --force-th 0.10 \
  --pred base

# VP extrap
python tools/genmap_split_agg.py \
  --glob-full "outputs/genmap/vp_extrap/*_traj_full.csv" \
  --glob-open "outputs/genmap/vp_extrap/*_traj_open.csv" \
  --out "outputs/genmap/vp_extrap/grid_metrics.csv" \
  --force-th 0.10 \
  --pred base

# VPF intrap
python tools/genmap_split_agg.py \
  --glob-full "outputs/genmap/vpf_intrap/*_traj_full.csv" \
  --glob-open "outputs/genmap/vpf_intrap/*_traj_open.csv" \
  --out "outputs/genmap/vpf_intrap/grid_metrics.csv" \
  --force-th 0.10 \
  --pred base

# VPF extrap
python tools/genmap_split_agg.py \
  --glob-full "outputs/genmap/vpf_extrap/*_traj_full.csv" \
  --glob-open "outputs/genmap/vpf_extrap/*_traj_open.csv" \
  --out "outputs/genmap/vpf_extrap/grid_metrics.csv" \
  --force-th 0.10 \
  --pred base

# A-3) ヒートマップ（4枚×4条件）＋差分（VPF−VP）
python tools/genmap_split_plot.py \
  --csv outputs/genmap/vp_intrap/grid_metrics.csv \
  --outdir outputs/genmap/vp_intrap/figs \
  --clip --annotate --annotate-outline --annotate-fmt "{:.3f}"

python tools/genmap_split_plot.py \
  --csv outputs/genmap/vp_extrap/grid_metrics.csv \
  --outdir outputs/genmap/vp_extrap/figs \
  --clip --annotate --annotate-outline --annotate-fmt "{:.3f}"

python tools/genmap_split_plot.py \
  --csv outputs/genmap/vpf_intrap/grid_metrics.csv \
  --outdir outputs/genmap/vpf_intrap/figs \
  --clip --annotate --annotate-outline --annotate-fmt "{:.3f}"

python tools/genmap_split_plot.py \
  --csv outputs/genmap/vpf_extrap/grid_metrics.csv \
  --outdir outputs/genmap/vpf_extrap/figs \
  --clip --annotate --annotate-outline --annotate-fmt "{:.3f}"


# 線形プローブ
# VP intrap
python tools/eval_probe.py \
  --dataset-root datasets/test_intrap \
  --ckpt outputs/ckpt_vp/best.pt \
  --probe outputs/probes/probe_vp.pt \
  --outdir outputs/probe_eval/vp_intrap \
  --level-plots \
  --plot-2d

# VP extrap
python tools/eval_probe.py \
  --dataset-root datasets/test_extrap \
  --ckpt outputs/ckpt_vp/best.pt \
  --probe outputs/probes/probe_vp.pt \
  --outdir outputs/probe_eval/vp_extrap \
  --level-plots \
  --plot-2d

# VPF intrap
python tools/eval_probe.py \
  --dataset-root datasets/test_intrap \
  --ckpt outputs/ckpt_vpf/best.pt \
  --probe outputs/probes/probe_vpf.pt \
  --use-force \
  --outdir outputs/probe_eval/vpf_intrap \
  --level-plots \
  --plot-2d

# VPF extrap
python tools/eval_probe.py \
  --dataset-root datasets/test_extrap \
  --ckpt outputs/ckpt_vpf/best.pt \
  --probe outputs/probes/probe_vpf.pt \
  --use-force \
  --outdir outputs/probe_eval/vpf_extrap \
  --level-plots \
  --plot-2d