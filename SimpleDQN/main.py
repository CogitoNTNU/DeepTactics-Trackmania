import torch
import time

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

t1 = torch.randn((6000, 6000), device=device)
t2 = torch.randn((6000, 6000), device=device)

print("mps")

for _ in range(5):
    start = time.perf_counter()
    t1 = t1 @ t2
    end = time.perf_counter()
    print(f"Time used: {end - start}")


t1 = torch.randn((6000, 6000), device="cpu")
t2 = torch.randn((6000, 6000), device="cpu")

print("cpu")
for _ in range(5):
    start = time.perf_counter()
    t1 = t1 @ t2
    end = time.perf_counter()
    print(f"Time used: {end - start}")
