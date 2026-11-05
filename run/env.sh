# Environment of the scripts that use qiskit_aer, in a pip .venv on Linux:   source run/env.sh
# qiskit_aer and torch each ship their own libgomp, and both have to be preloaded (otherwise: cannot allocate memory in
# static TLS block). The paths are absolute, so the scripts can run from any folder.
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LD_PRELOAD="$(ls "$_root"/.venv/lib/python3.*/site-packages/qiskit_aer.libs/libgomp*.so* | head -1):$(ls "$_root"/.venv/lib/python3.*/site-packages/torch/lib/libgomp.so* | head -1)"
export LD_PRELOAD
unset _root
