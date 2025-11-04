import time
import torch

# Choose device: CUDA > MPS > CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

n_tau = 5
taus = torch.rand(n_tau, device=device)

# Method 0: Original double loop
def original_loop():
    embedded_taus = []
    for tau in taus:
        embedded_tau = [torch.cos(tau * i * torch.pi) for i in range(64)]
        embedded_taus.append(torch.stack(embedded_tau))
    return torch.stack(embedded_taus)

# Method 1: Outer product
def outer_calc():
    co = torch.arange(64, device=device, dtype=taus.dtype) * torch.pi
    return torch.cos(torch.outer(taus, co))

# Method 2: Broadcasting
def broadcast_calc():
    i = torch.arange(64, device=device, dtype=taus.dtype) * torch.pi
    basis = i.unsqueeze(0)
    return torch.cos(taus.unsqueeze(1) * basis)

# Method 3: Einsum
def einsum_calc():
    i = torch.arange(64, device=device, dtype=taus.dtype) * torch.pi
    return torch.cos(torch.einsum("n,m->nm", taus, i))

# Method 4: List comprehension (stacked)
def listcomp_calc():
    i = torch.arange(64, device=device, dtype=taus.dtype) * torch.pi
    return torch.stack([torch.cos(tau * i) for tau in taus])

# Timing helper: run num_runs times and get average
def time_func(fn, name, num_runs=1000):
    # Warmup
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / num_runs
    print(f"{name}: shape={out.shape}, avg time per run={avg_time*1e3:.6f} ms")

# Run all methods
time_func(original_loop, "Original double loop")
time_func(outer_calc, "Outer product")
time_func(broadcast_calc, "Broadcasting")
time_func(einsum_calc, "Einsum")
time_func(listcomp_calc, "List comprehension")
