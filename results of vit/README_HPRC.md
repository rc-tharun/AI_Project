# Training on TAMU HPRC (Grace)

## Quick Start

### 1. Transfer files to Grace
```bash
scp train_safety_gate.py submit_grace.sh <NetID>@grace.hprc.tamu.edu:$SCRATCH/
```

### 2. Set up Kaggle credentials (for BDD100K download)
```bash
# On Grace:
mkdir -p ~/.kaggle
# Upload your kaggle.json from https://www.kaggle.com/settings
scp ~/.kaggle/kaggle.json <NetID>@grace.hprc.tamu.edu:~/.kaggle/
ssh <NetID>@grace.hprc.tamu.edu "chmod 600 ~/.kaggle/kaggle.json"
```

### 3. Submit the job
```bash
ssh <NetID>@grace.hprc.tamu.edu
cd $SCRATCH
sbatch submit_grace.sh
```

### 4. Monitor
```bash
squeue -u $USER          # check job status
tail -f safety_gate_*.out # watch output live
```

### 5. Get results
```bash
# Output files will be at $SCRATCH/safety_gate_output/
scp -r <NetID>@grace.hprc.tamu.edu:$SCRATCH/safety_gate_output/ ./results/
```

## Notes
- Uses 1x A100 GPU, ~4 hour wall time (usually finishes in ~1-2 hrs)
- Data stored at `$SCRATCH/data/raw/bdd100k/`
- Model checkpoints + plots saved to `$SCRATCH/safety_gate_output/`
- If A100 queue is long, change `--gres=gpu:a100:1` to `--gres=gpu:1` for any available GPU
