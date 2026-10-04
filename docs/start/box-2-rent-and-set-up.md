# 5b. Rent and set up the box

!!! abstract "In plain words"
    You pick a machine from vast.ai's list (a bit like choosing a rental car by size and price), start it, connect
    to it from your laptop's terminal, and run one script that installs everything recbench needs. The meter is
    running from the moment the machine starts, so have 5a finished first.

**Time:** about 30 minutes. **Cost:** about $0.30.

## Step 1: choose an offer

In the vast.ai console, open the page that lists machines to rent and set these filters:

| Filter | Choose | Why |
|---|---|---|
| GPU | 1 × 48 GB (RTX A6000, A40, L40S), or 2 × 24 GB | three GPU jobs share each GPU; EASE and Turbo-CF keep item × item matrices in GPU memory |
| CPU cores | 32 or more | 12 methods run on the CPU; each CPU worker gets 16–32 threads |
| RAM | 128 GB (64 GB at least) | a confirmation job loads up to 31 million events |
| Disk | 100 GB, set **before** you create the instance | the disk size cannot be changed afterwards |
| CUDA (driver) | 12.8 or higher | recent PyTorch builds need a recent driver |
| Reliability | 98% or higher | unreliable hosts disappear more often |
| Rental type | on-demand | "interruptible" is cheaper, but the machine can be taken back at any time |
| Template | PyTorch (Ubuntu 22.04 or 24.04), with SSH access | the NVIDIA driver and SSH come with it |

Sort by price and pick a cheap offer that meets all of these. Write down its **price per hour**; 5c uses it.

!!! info "Labels change"
    vast.ai renames filters and buttons from time to time. If a name above does not appear, look for the closest
    match (for example "GPU RAM" or "Max CUDA").

## Step 2: start it and copy the connection details

Create the instance and wait until its status says it is running (usually 1–5 minutes). Its **Connect** button shows
an SSH command such as:

```text
ssh -p 41234 root@203.0.113.7 -L 8080:localhost:8080
```

Turn that into a short name on your laptop. Add this block to `~/.ssh/config` (create the file if needed),
with your port and address:

```text
Host vast-gpu
    HostName 203.0.113.7
    Port 41234
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

`ServerAliveInterval` keeps an idle connection from being dropped. Every command in these pages uses `vast-gpu`.

## Step 3: connect for the first time

```bash
ssh vast-gpu
```

The first time, SSH asks whether you trust the machine's fingerprint. Type `yes`.

!!! success "You should see"
    A prompt on the box, such as `root@C.12345:~#`, and usually a green status bar at the bottom of the screen.
    That bar means you are inside **tmux**: vast.ai starts one for you. tmux keeps your programs running when the
    connection drops, which is exactly what a long run needs (5c explains the keys).

!!! warning "If SSH says the host identification has changed"
    That happens when a new rental reuses an address you connected to before. Remove the old fingerprint and
    connect again: `ssh-keygen -R "[203.0.113.7]:41234"`.

If you prefer to start tmux yourself, run `touch ~/.no_auto_tmux` once on the box and reconnect.

## Step 4: get the code and install everything

Check which disk is large: `df -h / /workspace 2>/dev/null`. Clone into the folder that has your 100 GB, usually
the home folder, or `/workspace` on some templates:

```bash
git clone https://github.com/trunghieu11/recbench.git
cd recbench
./scripts/setup_box.sh
```

`setup_box.sh` takes 5–10 minutes. It:

1. installs system packages: SuiteSparse and a C compiler for SANSA, tmux, rsync, and `htop` and `nvtop` for watching the machine;
2. creates `.venv` and installs recbench with the PyTorch build that matches this machine's driver, plus
   LightGBM, Optuna and sentence-transformers;
3. tries to install SANSA. If its build fails, setup goes on and SANSA's jobs later end `unsupported`;
4. downloads the text-embedding model once, and fetches GRU4Rec and the other third-party code at fixed commits;
5. records every package version in `runs/logs/pip-freeze-*.txt`.

!!! success "You should see"
    Near the end:

    ```text
    torch 2.14.1+cu128 | CUDA build: 12.8 | CUDA available: True | GPUs: ['NVIDIA RTX A6000']
    Sentence-transformer model cached.
    lightgbm 4.7.0 | optuna 5.0.0
    sansa ready
    ```

    The version numbers may differ. **`CUDA available: True` is the line that matters.** If PyTorch cannot use the
    GPU, the script stops and prints the driver's CUDA version with what to do: either rent a box whose driver
    supports CUDA 12.8 or higher, or reinstall PyTorch for this driver with the command it shows.

Check the GPU yourself:

```bash
nvidia-smi
```

!!! success "You should see"
    A table with your GPU's name, its memory (for example `0MiB / 49140MiB`) and, top right, the highest CUDA
    version the driver supports. No processes are listed yet.

**Next:** [5c. Run and monitor](box-3-run-and-monitor.md).
