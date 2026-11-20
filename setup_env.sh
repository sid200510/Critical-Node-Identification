#!/usr/bin/env bash
# Creates a Python environment for the critical-node experiments and registers it as a Jupyter kernel for VS Code.
# Usage (from the folder that contains this script and requirements.txt):
#     bash setup_env.sh
# Then in VS Code: open the .ipynb -> "Select Kernel" (top right) -> "Python (critical-nodes)".
set -e
ENV_DIR=".venv_critical"

python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip

# PyTorch, CPU-only build (no GPU on this machine; much smaller download than the CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# everything else the five experiment notebooks use
pip install -r requirements.txt

# make the environment selectable as a kernel in VS Code / Jupyter
python -m ipykernel install --user --name critical-nodes --display-name "Python (critical-nodes)"

python - <<'EOF'
import numpy, pandas, scipy, sklearn, matplotlib, torch
print("OK  numpy", numpy.__version__, "| pandas", pandas.__version__, "| scipy", scipy.__version__,
      "| scikit-learn", sklearn.__version__, "| matplotlib", matplotlib.__version__, "| torch", torch.__version__)
EOF
echo "Done. In VS Code choose the kernel 'Python (critical-nodes)'."
