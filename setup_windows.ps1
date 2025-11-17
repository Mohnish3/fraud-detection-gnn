# setup_windows.ps1
$env:VENV_DIR = ".\venv"
# Create venv if not exists
if (-Not (Test-Path $env:VENV_DIR)) {
    python -m venv $env:VENV_DIR
    Write-Host "Virtualenv created at $env:VENV_DIR"
} else {
    Write-Host "Virtualenv already exists at $env:VENV_DIR"
}

# Activate venv for this script
& $env:VENV_DIR\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install core dependencies first
pip install numpy pandas networkx scikit-learn matplotlib tqdm

# Install torch. Adjust if you need CUDA; this is CPU fallback.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install PyG CPU wheels (the recommended way is to follow PyG website for your torch/cuda combo).
# Try CPU wheels (should work on Windows for testing). If your setup has CUDA, follow PyG docs instead.
pip install torch-geometric --no-deps

# Finally install remaining (safety)
pip install -r requirements.txt --no-deps

Write-Host "Setup complete. Activate the venv using: .\venv\Scripts\Activate.ps1"
