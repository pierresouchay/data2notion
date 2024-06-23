#!/bin/sh
set -eu
isort src
ruff format
mypy src --install-types
flake8 --max-line-length 120 src
