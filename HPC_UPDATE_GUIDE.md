# HPC File Update Commands

## Method 1: Transfer File from Local to HPC (Recommended)

### Using WinSCP (GUI - Easiest):
1. Open WinSCP
2. Connect to your HPC (host: your-hpc-address, username: mpi)
3. Navigate to `/home/mpi/Sharon/IIT/Core/`
4. Drag and drop `train.py` from your local `d:\IIT\Core\` folder
5. Click "Yes" to overwrite

### Using SCP Command (Windows PowerShell):
```powershell
# From your Windows machine
scp d:\IIT\Core\train.py mpi@your-hpc-address:/home/mpi/Sharon/IIT/Core/train.py
```

## Method 2: Edit Directly on HPC (via PuTTY)

```bash
# In PuTTY terminal
cd ~/Sharon/IIT/Core

# Backup old file first
cp train.py train.py.backup

# Edit with nano (easier for beginners)
nano train.py

# Key shortcuts for nano:
# - Ctrl+O: Save file
# - Ctrl+X: Exit
# - Ctrl+K: Cut line
# - Ctrl+U: Paste line

# OR edit with vim (more powerful)
vim train.py
# Press 'i' to enter insert mode
# Make changes
# Press Esc, then type :wq to save and quit
```

## Method 3: Clone from Git (Best Practice)

```bash
# On HPC
cd ~/Sharon/IIT
git pull origin main  # If you have git set up

# Or if not using git yet:
git init
git remote add origin https://github.com/yourusername/yourrepo.git
git add .
git commit -m "Updated train.py for dual datasets"
git push -u origin main
```

## After Updating, Run Training:

```bash
# Navigate to Core directory
cd ~/Sharon/IIT/Core

# Activate virtual environment (if you have one)
# source venv/bin/activate

# Run training with GPU
python3 train.py --epochs 100 --batch_size 32 --learning_rate 1e-4

# Or submit as HPC job (if using SLURM)
sbatch run_training.sh
```

## Verify GPU is Working:

```bash
python3 << 'EOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
EOF
```

## Check Training Progress:

```bash
# Monitor GPU usage
nvidia-smi

# Watch training output
tail -f nohup.out  # If running with nohup

# Check TensorBoard logs
tensorboard --logdir=runs --host=0.0.0.0 --port=6006
```
