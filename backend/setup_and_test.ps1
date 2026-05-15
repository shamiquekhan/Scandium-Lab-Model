python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pytest pytest-asyncio httpx
python -m pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
python -m pip install -r requirements.txt
pytest tests/
