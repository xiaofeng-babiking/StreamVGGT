<div align="center">
<h1>Streaming 4D Visual Geometry Transformer</h1>
</div>

### [Paper](https://arxiv.org/abs/2507.11539)  | [Project Page](https://wzzheng.net/StreamVGGT)  | [Online Demo](https://huggingface.co/spaces/lch01/StreamVGGT)

>Streaming 4D Visual Geometry Transformer

>Dong Zhuo<sup>\*</sup>, [Wenzhao Zheng](https://wzzheng.net/)<sup>*</sup>$\dagger$,  Jiahe Guo, Yuqi Wu, [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/)

<sup>*</sup> Equal contribution. $\dagger$ Project leader.


**StreamVGGT**, a causal transformer architecture for **real-time streaming 4D visual geometry perception** compatiable with LLM-targeted attention mechanism (e.g., [FlashAttention](https://github.com/Dao-AILab/flash-attention)), delivers both fast inference and high-quality 4D reconstruction.

## News

- **[2025/7/18]** [Demo](https://huggingface.co/spaces/lch01/StreamVGGT) and [checkpoints](https://huggingface.co/lch01/StreamVGGT/) released on Hugging Face; demo code is available for local launch.
- **[2025/7/15]** Paper released on [arXiv](https://arxiv.org/abs/2507.11539).
- **[2025/7/14]** Release the code for **fine-tuning VGGT**.
- **[2025/7/13]** Check out [Point3R](https://github.com/YkiWu/Point3R) for another streaming 3D reconstruction work of ours!
- **[2025/7/13]** Distillation code for VGGT is released.
- **[2025/7/13]** Inference code with [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) is released.
- **[2025/7/13]** Training/evaluation code release.

## Overview

Given a sequence of images, unlike offline models that require reprocessing the entire sequence and reconstructing the entire scene upon receiving each new image, our StreamVGGT employs temporal
causal attention and leverages cached memory token to support efficient incremental on-the-fly reconstruction, enabling interative and real-time online applitions.

<img src="./assets/teaser_v2_01.png" alt="overview" style="width: 100%;" />

### On-the-Fly Online Reconstruction from Streaming Inputs

<img src="./assets/results.png" alt="overview" style="width: 100%;" />

### Installation

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. Install it first:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Clone StreamVGGT
```bash
git clone https://github.com/wzzheng/StreamVGGT.git
cd StreamVGGT
```

2. Install system-level build tools (cmake + OpenMP runtime) — these are not Python packages, so uv does not manage them. With conda:
```bash
conda install -c conda-forge 'cmake>=3.14' 'llvm-openmp<16'
```
Or with apt on Debian/Ubuntu:
```bash
sudo apt-get install -y cmake libomp-dev
```

3. Sync Python dependencies (PyTorch comes from the `cu121` wheel index, configured in `pyproject.toml`):
```bash
uv sync
```
This creates a `.venv/` in the project root and installs all core deps from `uv.lock`. To work inside it, either prefix commands with `uv run` or activate it with `source .venv/bin/activate`.

To also install optional dependency groups:
```bash
uv sync --extra demo          # for demo_gradio.py
uv sync --extra pose-eval     # for camera pose evaluation
uv sync --extra orchestrator  # for the multi-node training orchestrator (asyncssh + click + rich)
uv sync --all-extras          # everything
```

### Download Checkpoints
Please download pretrained teacher model from [here](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt).

The checkpoint of StreamVGGT is also available at both [Hugging Face](https://huggingface.co/lch01/StreamVGGT/) and [Tsinghua cloud](https://cloud.tsinghua.edu.cn/d/d6ad8f36fcd541bcb246/).


## Data Preparation
### Training Datasets
Our training data includes 14 datasets. Please download the datasets from their official sources and refer to [CUT3R](https://github.com/CUT3R/CUT3R/blob/main/docs/preprocess.md) for processing these datasets.

  - [ARKitScenes](https://github.com/apple/ARKitScenes) 
  - [BlendedMVS](https://github.com/YoYo000/BlendedMVS)
  - [CO3Dv2](https://github.com/facebookresearch/co3d)
  - [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/)
  - [MVS-Synth](https://phuang17.github.io/DeepMVS/mvs-synth.html)
  - [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) 
  - [ScanNet](http://www.scan-net.org/ScanNet/)
  - [Spring](https://spring-benchmark.org/)
  - [Hypersim](https://github.com/apple/ml-hypersim)
  - [WildRGB-D](https://github.com/wildrgbd/wildrgbd/)
  - [WayMo Open dataset](https://github.com/waymo-research/waymo-open-dataset)
  - [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
  - [OmniObject3D](https://omniobject3d.github.io/)
  - [PointOdyssey](https://pointodyssey.com/)

### Evaluation Datasets
Please refer to [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to prepare Sintel, Bonn, KITTI, NYU-v2, ScanNet, 7scenes and Neural-RGBD datasets.

## Folder Structure
The overall folder structure should be organized as follows：
```
StreamVGGT
├── ckpt/
|   ├── model.pt
|   └── checkpoints.pth
├── config/
|   ├── train.yaml
|   ├── train_smoke.yaml          # smoke variant used by L4 multi-node validation
|   └── finetune.yaml
├── data/
│   ├── eval/
|   |   ├── 7scenes
|   |   ├── bonn
|   |   ├── kitti
|   |   ├── neural_rgbd
|   |   ├── nyu-v2
|   |   ├── scannetv2
|   |   └── sintel
│   ├── train/
│   │   ├── processed_arkitscenes
|   |   ├── ...
├── scripts/                      # multi-node orchestrator entrypoints
|   ├── bootstrap_venv.sh
|   ├── svggt-orch                # CLI wrapper (PYTHONPATH=src, exec venv python)
|   ├── start_into_docker.sh
|   ├── worker_entry.sh           # docker CMD on each worker
|   └── lib/docker_env.sh         # shared docker run flags
├── workflows/                    # orchestrator state + docs
|   ├── cluster.yaml              # declarative cluster config (edit me)
|   ├── cluster.schema.json
|   ├── nodes.json                # generated by `discover` (gitignored)
|   ├── jobs/<job_id>/            # per-launch manifest + logs (gitignored)
|   └── train.html                # step-by-step operator walkthrough
└── src/
    ├── svggt_orch/               # orchestrator package (Python)
    └── ...
```

## Finetuning VGGT
We also provide the following commands to fine-tune VGGT (excluding the track head) if you like. 
```bash
cd src/
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu --main_process_port 26902 ./finetune.py --config-name finetune
```

## Training StreamVGGT
We provide the following commands for training.

```bash
cd src/
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu --main_process_port 26902 ./train.py --config-name train
```

## Multi-Node Distillation Training

For runs that span multiple A100/H100 nodes sharing a single `/jfs` (JuiceFS) filesystem, this repo ships a Python orchestrator (`src/svggt_orch/`) that:

- Discovers idle GPUs across a pool of SSH-reachable hosts.
- Fans out `docker run` invocations with the right `--machine_rank`, `MAIN_PROCESS_IP`, and NCCL env to each node.
- Streams per-rank logs back to the head node with a `rich`-colored demux.
- Renders a live TUI dashboard of GPU utilization, IB throughput, and heartbeat freshness.
- Aborts atomically: if any of the N `docker run` calls fails, the orchestrator tears down every started container (no silent reduced world-size training).

The complete operator walkthrough — including architecture diagrams, every CLI subcommand's expected output, and an L4 smoke checklist — is at [`workflows/train.html`](./workflows/train.html). Open it from a browser via `file://...`. The quick start below covers the common case.

### Prerequisites

- `uv` available on the head node (the same install used by `uv sync` above).
- SSH key trust from head → every worker (`asyncssh` uses `known_hosts=None`, so trust-on-first-use is OK, but the key itself must work).
- `/jfs` mounted on every node at the same path.
- Docker daemon running on every worker; the base image
  `babiking/ubuntu:12.8.1-cudnn-devel-ubuntu24.04` pulled (orchestrator does not pre-pull).
- Mellanox IB `mlx5_1` ACTIVE on every node (or update `workflows/cluster.yaml#nccl.ib_hca` to match your hardware).

### Step 1 — Bootstrap the shared venv (once)

`uv sync --extra orchestrator` populates `<repo>/.venv` on `/jfs`, shared by all nodes:

```bash
./scripts/bootstrap_venv.sh
```

The script is idempotent, `flock`-protected against concurrent peer invocations, and refuses to run if `/jfs` has less than 5 GB free.

### Step 2 — Edit your cluster manifest

`workflows/cluster.yaml` is the **single source of truth** for the orchestrator. Put your worker hostnames or IPs in `candidate_hosts`:

```yaml
ssh:
  user: jing.feng
  # Pick at least one auth method (asyncssh tries key first, then password):
  identity_file: ~/.ssh/id_ed25519          # SSH key (preferred)
  # password: "${SSH_PASSWORD}"             # env-var interpolation (no plaintext in git)
  # password: "my-plaintext-pw"             # literal — warns at load
  connect_timeout_s: 5

candidate_hosts:                 # hostnames OR IPs work; mix freely
  - gpu003
  - 172.31.208.7
  - 172.31.208.8

docker:
  image: babiking/ubuntu:12.8.1-cudnn-devel-ubuntu24.04
  shm_size: 16G

paths:
  repo: /jfs/jing.feng/codebases/StreamVGGT
  venv: /jfs/jing.feng/codebases/StreamVGGT/.venv
  jfs: /jfs

discovery:
  idle_threshold_mib: 500        # GPU counts as idle if memory.used < this
  min_idle_gpus_per_node: 1      # skip nodes with fewer idle GPUs

nccl:
  ib_hca: mlx5_1                 # IB HCA for NCCL data plane
  socket_ifname: bond0           # control plane interface
  ib_disable: 0
  async_error_handling: 1        # critical: makes one-rank deaths abort all peers in seconds
  debug: WARN                    # INFO during bring-up, WARN once stable
  ib_timeout: 23
  net_gdr_level: PIX

train:
  entrypoint: src/train.py
  config_name: train             # or train_smoke for L4 validation
  main_port: 26902
```

Validate it against the schema:

```bash
uv run --extra orchestrator --group dev check-jsonschema \
    --schemafile workflows/cluster.schema.json workflows/cluster.yaml
```

### Step 3 — Discover idle GPUs

```bash
./scripts/svggt-orch discover --dry-run    # prints the SSH plan, no actual connect
./scripts/svggt-orch discover              # real fan-out; writes workflows/nodes.json
```

A GPU counts as idle when `nvidia-smi --query-gpu=memory.used` reports less than `idle_threshold_mib` MiB. Unreachable hosts are skipped silently; hosts with fewer than `min_idle_gpus_per_node` free GPUs are dropped.

Inspect the result:

```bash
cat workflows/nodes.json
# [
#   {"host": "gpu003", "bond_ip": "172.31.208.5", "ib_ip": "10.10.100.5",
#    "free_gpus": [0,1,2,3,4,5], "detected_at": "..."},
#   ...
# ]
```

### Step 4 — Launch the job

```bash
./scripts/svggt-orch launch --dry-run --exp-name StreamVGGT_alpha0.1_lr1e-5_nog
./scripts/svggt-orch launch --exp-name StreamVGGT_alpha0.1_lr1e-5_nog
```

The launcher:

- Reads `workflows/nodes.json`, sums `free_gpus` across nodes for `NUM_PROCESSES` (world size).
- Picks `nodes[0].ib_ip` as `MAIN_PROCESS_IP` so NCCL bootstrap lands on IB.
- Writes `workflows/jobs/<job_id>/manifest.json` before spawning anything.
- Fan-outs `docker run -d --name svggt-<job_id> --gpus ... babiking/... /workspace/scripts/worker_entry.sh` to every node via asyncssh.
- If **any** node's `docker run` fails, fans out `docker rm -f` to every node and exits 3.

### Step 5 — Monitor

Three layers of visibility, all driven off the manifest written in step 4. Use them in separate terminals:

```bash
./scripts/svggt-orch tail    --job <job_id>    # per-rank log demux with rich colors
./scripts/svggt-orch monitor --job <job_id>    # live TUI: GPU util/mem/temp + IB GB/s
./scripts/svggt-orch tb      --job <job_id>    # spawn TensorBoard against checkpoints/<exp>/logs
```

`train.py` writes a `<exp>/heartbeat` file every `print_freq` steps on rank 0; the `monitor` TUI flips its banner red if the mtime goes stale (default threshold 600 s). This catches silent hangs without waiting out the 6000 s NCCL timeout.

### CLI surface

| Command | Purpose | Exit codes |
|---|---|---|
| `./scripts/bootstrap_venv.sh` | Sync `.venv` via uv | 0 ok / 1 config / 2 ENOSPC / 3 flock |
| `svggt-orch discover [--dry-run]` | SSH-fan-out GPU scan → `workflows/nodes.json` | 0 ok / 2 no nodes |
| `svggt-orch launch [--dry-run] [--exp-name X]` | Fan-out `docker run` with atomic abort | 0 ok / 2 missing nodes.json / 3 launch failed |
| `svggt-orch tail --job <id>` | Per-rank `docker logs -f` demux | 0 ok / 1 no manifest |
| `svggt-orch monitor --job <id>` | Live TUI dashboard | 0 ok / 1 no manifest |
| `svggt-orch tb --job <id>` | Spawn TensorBoard with SSH-forward hint | 0 ok / 1 no manifest / no `tensorboard` |
| `svggt-orch status --job <id>` | Manifest summary + per-host docker state | 0 ok / 1 no manifest |
| `svggt-orch kill --job <id> --yes` | Fan-out `docker rm -f`; flip manifest to "killed" | 0 ok / 1 no manifest |

### Recovery & resume

`train.py` auto-resumes from `checkpoints/<exp_name>/checkpoint-last.pth` when present (see `train.py:137-138`), so the kill-and-relaunch loop loses at most one `save_freq` interval of progress:

```bash
./scripts/svggt-orch kill    --job <old_id> --yes
./scripts/svggt-orch discover                 # refresh idle GPUs
./scripts/svggt-orch launch  --exp-name StreamVGGT_alpha0.1_lr1e-5_nog
# train.py logs: "Resuming from checkpoint-last.pth"
```

### L4 smoke validation

Before relying on the orchestrator for a long run, exercise the full path with a tiny config:

```bash
./scripts/svggt-orch launch --exp-name StreamVGGT_smoke
# in cluster.yaml, set train.config_name: train_smoke for ~20 steps × 100 samples
```

The end-to-end checklist (10 steps including crash injection and heartbeat staleness) is the last section of [`workflows/train.html`](./workflows/train.html).

## Evaluation
The evaluation code follows [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md), [CUT3R](https://github.com/CUT3R/CUT3R/blob/main/docs/eval.md) and [VGGT](https://github.com/facebookresearch/vggt).

```bash
cd src/
```
### Monodepth
```bash
bash eval/monodepth/run.sh 
```

Results will be saved in `eval_results/monodepth/${data}_${model_name}/metric.json`.

### VideoDepth
```bash
bash eval/video_depth/run.sh 
```

Results will be saved in `eval_results/video_depth/${data}_${model_name}/result_scale.json`.

### Multi-view Reconstruction
```bash
bash eval/mv_recon/run.sh 
```

Results will be saved in `eval_results/mv_recon/${model_name}_${ckpt_name}/logs_all.txt`.

### Camera Pose Estimation
1. Install the required dependencies (`pycolmap` and `pyceres` are in the `pose-eval` extra; LightGlue is a git checkout):
```bash
uv sync --extra pose-eval
git clone https://github.com/cvg/LightGlue.git
uv pip install -e ./LightGlue
```
2. Please refer to [VGGT](https://github.com/facebookresearch/vggt) to prepare the co3d dataset.

3. Run the evaluation code:
```bash
python eval/pose_evaluation/test_co3d.py --co3d_dir /YOUR/CO3D/PATH --co3d_anno_dir /YOUR/CO3D/ANNO/PATH --seed 0
```

## Demo
We provide a demo for StreamVGGT, based on the demo code from [VGGT](https://github.com/facebookresearch/vggt). You can follow the instructions below to launch it locally or try it out directly on [Hugging Face](https://huggingface.co/spaces/lch01/StreamVGGT).
```bash
uv sync --extra demo
uv run python demo_gradio.py
```

**Note**: While StreamVGGT typically reconstructs a scene in under one second, 3D point visualization may take much longer due to slower third-party rendering.

## Acknowledgements
Our code is based on the following brilliant repositories:

[DUSt3R](https://github.com/naver/dust3r)
[MonST3R](https://github.com/Junyi42/monst3r.git)
[Spann3R](https://github.com/HengyiWang/spann3r.git)
[CUT3R](https://github.com/CUT3R/CUT3R)
[VGGT](https://github.com/facebookresearch/vggt)
[Point3R](https://github.com/YkiWu/Point3R)

Many thanks to these authors!

## Citation

If you find this project helpful, please consider citing the following paper:
```
@article{streamVGGT,
      title={Streaming 4D Visual Geometry Transformer}, 
      author={Dong Zhuo and Wenzhao Zheng and Jiahe Guo and Yuqi Wu and Jie Zhou and Jiwen Lu},
      journal={arXiv preprint arXiv:2507.11539},
      year={2025}
}
```
