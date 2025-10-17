# RUNNING PHARAGLOW IN JUPYTERLAB NOTEBOOK

This guide assumes you have been given access to `soma HPC cluster` and you are able to submit jobs to the CPU/GPU-interactive partitions.

## STEPS:

## 1) Log into soma cluster

The first step is to login into soma HPC cluster via ssh.  Our recommendation for new users is that they install and use the [Mobaxterm Portable SSH](https://mobaxterm.mobatek.net/download.html) client.

Start a new SSH session using their Active Directory user/password. (same as in Windows)

## 2) Clone pharaglow


```bash
git clone https://github.com/scholz-lab/PharaGlow.git pharaglow
```

**NOTE:** To clone the pharaglow repository I will recommend to first configure your `~/.netrc` file. It should be still possible to clone the repository without it, but you will not be able to push/contribute your local changes. 

1) You need first to generate a PAT (Personal Access Token) in _GitHub_ (Settings/Developer Settings).
2) Create a file with the name `.netrc` inside your home directory. The file should have the following format:
    ```bash
    machine github.com login YOUR_GITHUB_USER password GITHUB_PERSONAL_ACCESS_TOKEN 
    ```


## 3) Allocate compute node

Next step is to allocate compute resources from one of the interactive compute partitions. The _soma_ HPC cluster has two partitions that are exclusively reserved for interactive usage:

```bash
somalogin01 ~ $ sinfo -s -p GPU-interactive,CPU-interactive
PARTITION       AVAIL  TIMELIMIT   NODES(A/I/O/T) NODELIST
CPU-interactive    up 14-00:00:0        9/11/0/20 somacpu[201-220]
GPU-interactive    up 14-00:00:0        2/18/0/20 somagpu[201-220]
```

* <details>
    <summary><code>CPU-interactive</code> partition:</summary>

    + Partition has 20 compute nodes. 
    + Node hardware configuration:
        ```bash
        somalogin01 ~ $ scontrol show node somacpu201
        NodeName=somacpu201 Arch=x86_64 CoresPerSocket=20 
        RealMemory=358400 AllocMem=0 FreeMem=368349 Sockets=2 Boards=1
        Partitions=CPU-interactive 
        CfgTRES=cpu=40,mem=350G
        ```
</details>

```bash
srun -p CPU-interactive -t 8:00:00 --mem 32G --cpus-per-task=8 -J jupyterlab --pty bash
``` 


* <details>
    <summary><code>GPU-interactive</code> partition:</summary>

    + Partition has 20 compute nodes.
    + Node hardware configuration:
        ```bash
        somalogin01 ~ $ scontrol show node somagpu201
        NodeName=somagpu201 Arch=x86_64 CoresPerSocket=10 
        Gres=gpu:nvidia_geforce_rtx_2080_ti:1(S:1)
        RealMemory=179200 AllocMem=0 FreeMem=179795 Sockets=2 Boards=1
        Partitions=GPU-interactive 
        CfgTRES=cpu=20,mem=175G,gres/gpu=1,gres/gpu:nvidia_geforce_rtx_2080_ti=1
        ```
</details>

```bash
srun -p GPU-interactive -t 8:00:00 --mem 0 --cpus-per-task=20 -J jupyterlab --gres=gpu:1 --pty bash
``` 
## 4) Start jupyterlab

Once the node is allocated via `slurm` the interactive session starts and the user is automatically logged into the node inside a `bash` terminal. 

The next step is to change into pharaglow, activate the `pumping` conda environment and start the jupyterlab server in a port of our choice.

```bash
cd pharaglow
conda activate pumping
jupyter-lab --port=6060 --ip="$(hostname -i)" --no-browser
```
## 5) Create ssh-tunnel

At this point `jupyter-lab` is running on the remote compute node. We would like to be able to access it locally in our client using the browser. Therefore we need to create an ssh-tunnel that connects from our local client to the remote server via soma login node (eg. `somalogin01`).

```bash
ssh -fN -L 6060:somacpu2XX:6060 somalogin01
```

## 6) Open jupyterlab link using browser

Finally we can open a new browser window and enter the link we obtained when we launch `jupyterlab` (The one using the `localhost`). You can navigate from there to the `notebooks` directory and run the cells of the [notebooks/PharaGlowHPC.ipynb](notebooks/PharaGlowHPC.ipynb) notebook.