# Google TPU Setup Guide

## Project Info
- Project ID: `diffusion-mech-interp`
- Project Number: `17993615303`
- Account: `trevorbchen@gmail.com`
- Fellowship: 30-day TRC (started ~April 2026)

## Quota
| Type | Chips | Zone |
|------|-------|------|
| v4 on-demand | 32 | us-central2-b |
| v5e spot | 64 | us-central1-a |
| v4 spot | 32 | us-central2-b |
| v6e spot | 64 | us-east1-d |
| v5e spot | 64 | europe-west4-b |
| v6e spot | 64 | europe-west4-a |

## gcloud CLI Location
```
C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd
```

In bash, use:
```bash
GCBIN="C:\\Program Files (x86)\\Google\\Cloud SDK\\google-cloud-sdk\\bin\\gcloud.cmd"
```

## Create a TPU VM

### On-demand (won't get preempted, but capacity not always available)
```bash
"$GCBIN" compute tpus tpu-vm create diffusion-interp \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-ubuntu2204-base
```

### Spot (cheaper quota, but can get preempted mid-job)
```bash
"$GCBIN" compute tpus tpu-vm create diffusion-interp \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-ubuntu2204-base \
  --spot
```

If a zone has no capacity, try another zone (see quota table above).
For v5e/v6e use `v5litepod-4` or `v6e-4` as accelerator-type and appropriate version strings.

Creation takes 2-5 minutes.

## SSH into TPU VM

SSH key is at `~/.ssh/google_compute_engine`. Username is `Trevo`.

```bash
# Get the IP
IP=$("$GCBIN" compute tpus tpu-vm describe diffusion-interp \
  --zone=us-central2-b \
  --format="value(networkEndpoints[0].accessConfig.externalIp)" 2>/dev/null)

# SSH in
ssh -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null Trevo@$IP
```

If SSH key is rejected (new VM), push keys to project metadata first:
```bash
echo "Trevo:$(cat ~/.ssh/google_compute_engine.pub)" > /tmp/ssh_keys.txt
"$GCBIN" compute project-info add-metadata --metadata-from-file=ssh-keys=/tmp/ssh_keys.txt
```

## Install Python deps on the VM

```bash
ssh ... Trevo@$IP "pip install 'torch~=2.6.0' 'torch_xla[tpu]~=2.6.0' diffusers Pillow \
  -f https://storage.googleapis.com/libtpu-releases/index.html"
```

**Important**: torch and torch_xla versions must match. 2.6.0 works as of April 2026.

## Upload and run a script

```bash
# Upload
scp -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  my_script.py Trevo@$IP:~/my_script.py

# Run with nohup (survives SSH disconnect)
ssh ... Trevo@$IP "nohup python3 ~/my_script.py > ~/run.log 2>&1 &"

# Check progress
ssh ... Trevo@$IP "tail -20 ~/run.log"

# Download results
scp -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  Trevo@$IP:~/output.png ./output.png
```

## TPU vs CPU on the VM

torch_xla compiles lazily and recompiles whenever the computation graph shape changes.
For loops with `.item()` calls (like Jacobian finite differences), this causes extreme slowness.

**Use TPU for**: batched forward passes, training loops, standard sampling
**Use CPU for**: Jacobian computation (finite difference loops with .item()), anything with dynamic shapes

To force CPU in code:
```python
DEVICE = torch.device("cpu")
# NOT: xm.xla_device()
```

The TPU VM has 400GB RAM and a fast Xeon CPU, so CPU-mode is still useful
(faster than a laptop for large workloads).

## Check status
```bash
"$GCBIN" compute tpus tpu-vm list --zone=us-central2-b
```

States: CREATING, READY, PREEMPTED, DELETING

## Delete when done (optional)
```bash
"$GCBIN" compute tpus tpu-vm delete diffusion-interp \
  --zone=us-central2-b --quiet
```

## Cost
- TPU chips: **free** (TRC fellowship)
- Boot disk (~100GB): ~$4/month, prorated by hour
- Network: ~$0.12/GB egress, negligible for small files
- $300 intro credit covers everything easily
- Fine to leave a VM running — it costs pennies per hour idle
- Spot instances are free but get preempted (killed) without warning

## Lessons learned
- On-demand in us-central2-b sometimes has no capacity — try later or another zone
- Spot instances get preempted fast — always use `nohup` and save checkpoints
- The gcloud SSH subcommand breaks in bash because of spaces in the gcloud path — use direct SSH with the google_compute_engine key instead
- torch_xla is terrible for Jacobian finite-difference loops — just use CPU on the VM
