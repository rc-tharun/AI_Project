#!/bin/bash
#SBATCH --job-name=safety_gate
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ─── Load modules on Grace ─────────────────────────────────
module purge
module load GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2-CUDA-12.1.1
module load scikit-learn/1.4.0
module load matplotlib/3.8.2

# ─── Run training ─────────────────────────────────────────
echo "Starting training on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python train_safety_gate.py

echo "Finished at $(date)"
echo "Output saved to: $SCRATCH/safety_gate_output/"
