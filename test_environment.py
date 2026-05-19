import torch, importlib, sys

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
try:
    import triton
    print("triton:", triton.__version__)
except Exception as e:
    print("triton import failed:", e)

# inspect mamba_ssm op module
try:
    import mamba_ssm.ops.triton.ssd_combined as ssd
    # try to see the symbol name used by the op forward wrapper
    names = [n for n in dir(ssd) if "causal" in n.lower() or "conv1d" in n.lower()]
    print("ssd_combined has names:", names[:20])

    # check actual function pointer referenced inside the forward wrapper (best-effort)
    # Many implementations assign the compiled kernel into a python variable; try common names:
    for cand in ("causal_conv1d_fwd_function", "causal_conv1d", "conv1d_fwd_function"):
        if hasattr(ssd, cand):
            print(cand, "is", getattr(ssd, cand))
            print("callable?", callable(getattr(ssd, cand)))
except Exception as e:
    print("Error inspecting mamba_ssm.ops.triton.ssd_combined:", repr(e))

# print mamba_ssm version if available
try:
    import mamba_ssm
    print("mamba_ssm:", mamba_ssm.__version__ if hasattr(mamba_ssm, "__version__") else "unknown")
except Exception as e:
    print("mamba_ssm import failed:", e)

# quick sanity: ensure CUDA device is available
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
