#!/bin/sh
pip install termcolor
pip install paddlepaddle
pip install paddleocr
pip install gdown
pip uninstall protobuf -y
pip install --no-binary protobuf protobuf
pip install onnxruntime
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install easyocr