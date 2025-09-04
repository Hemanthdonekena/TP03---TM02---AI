#!/usr/bin/env bash
set -euo pipefail
python -c "from src.train import train_all; train_all('data')"
echo 'Models saved under data/'
