# Geometric and 3D Computer Vision project
Author: Alvise Favero (888851@stud.unive.it)
Academic Year: 2025/2026

## Running
The project requires Python 3.12 because Open3D doesn't support newer versions.
I recommend using uv to change Python version easily.

Running with uv:
```sh
# Create venv and install dependencies
uv venv
source .venv/bin/activate
uv sync

# Run application
uv run main.py [K.txt] [dist.txt] [video]
```

With a normal Python 3.12 installation:
```sh
# Create venv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Run application
python3 main.py [K.txt] [dist.txt] [video]
```
