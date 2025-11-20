
# 1. Install PyTorch (Manual command for CUDA 12.6)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# 2. Install dependencies
# (Make sure your requirements.txt includes 'numpy<2.0.0'!)
pip install -r requirements.txt

# 3. Install basicsr
# We use 'pip install -e .' with '--no-build-isolation'.
pip install -e . --no-build-isolation

echo "Miniconda and virtual environment installed successfully!"
