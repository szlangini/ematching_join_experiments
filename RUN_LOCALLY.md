# Running the benchmarks on your own machine

There are two notebooks:

| notebook | what it runs | needs |
|---|---|---|
| `laptop_ematching_bench.ipynb` | CPU verticals + DuckDB + DataFusion | just Python + pip (any OS) |
| `colab_ematching_gpu.ipynb` | the above **plus** the GPU (CUDA) verticals | NVIDIA GPU + **CUDA Toolkit** |

If you have a GPU **and** a real CPU, the second notebook is the one you want — it
gives a GPU-vs-CPU comparison on your *own* hardware (much fairer than Colab,
whose CPU is a weak 2-vCPU VM).

---

## 0. Get the files

From Colab: **File → Download → Download .ipynb**. Also grab **`requirements.txt`**
(in this repo). Put the `.ipynb`, `requirements.txt` in the same folder.

## 1. Python + packages (needed for both notebooks)

Python 3.10+ (3.11 / 3.12 recommended).

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
#  .venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
```

Open a notebook with `jupyter lab` (then click it), or in VS Code (Python +
Jupyter extensions). Then **Run all**.

## 2. First, the CPU-only notebook (works everywhere)

Run `laptop_ematching_bench.ipynb` → **Run all**. No GPU, no compiler. If this
works, your Python stack is good and you've isolated any remaining issue to the
GPU toolchain.

## 3. The GPU notebook — install the CUDA Toolkit

It compiles CUDA with `nvcc` and runs the binary, so you need two things:

- **NVIDIA driver** (you have it). Verify: `nvidia-smi`.
- **CUDA Toolkit** = provides `nvcc`. Verify: `nvcc --version`.

### Linux
Install the toolkit (pick one):
- `sudo apt install nvidia-cuda-toolkit`  *(simplest; may be an older CUDA)*
- conda, no sudo: `conda install -c nvidia cuda-toolkit`
- official: <https://developer.nvidia.com/cuda-downloads>

Then check `nvcc --version` **and** `nvidia-smi` both work, open
`colab_ematching_gpu.ipynb`, **Run all**.

### Windows
The GPU notebook's shell cells are written for Linux (`./ematching_gpu`, `||`,
line-continuation `\`). Two ways:

- **Recommended — WSL2.** Install WSL2 (Ubuntu), install the NVIDIA *CUDA-on-WSL*
  toolkit, then follow the **Linux** steps above *inside WSL*. Cleanest, no cell
  edits.
- **Native Windows.** Needs the CUDA Toolkit **and** the MSVC C++ compiler
  (Visual Studio Build Tools — `nvcc` uses `cl.exe`), plus editing the
  compile/run cells to Windows syntax (`ematching_gpu.exe`, no bash `||`/`\`).
  Ask and I'll provide a Windows-native version of those two cells.

## Common errors → fixes

| symptom | cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'google.colab'` | old notebook copy. Re-download — the save cell now skips Colab-only code when run locally. |
| `nvcc: command not found` | CUDA Toolkit not installed / not on PATH (Section 3). |
| `./ematching_gpu: No such file or directory` (Windows) | use WSL2, or ask for the Windows-native cells. |
| `pip install` fails | use Python 3.10+ and make sure the venv is activated. |
| GPU cells fine but slow / OOM at N=8M | lower the first arg of `!./ematching_gpu 8388608 32768` (e.g. `2097152`). |

## Still stuck?

Send: your **OS** (`uname -a`, or Windows version), the **exact error text**, and
the output of `nvidia-smi` and `nvcc --version`. That pins it down immediately.
