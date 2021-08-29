import numpy as np
import torch
import random
import time
import pandas as pd
import matplotlib.pyplot as plt

def gen_idx_rand(num_points, num_samples):
    return torch.Tensor(random.sample(range(num_points), num_samples)).to('cuda')

def gen_idx_mult_cpu(num_points, num_samples):
    p = torch.ones(num_points) / num_points
    return p.multinomial(num_samples=num_samples, replacement=False).to('cuda')

def gen_idx_mult_gpu(num_points, num_samples):
    p = torch.ones(num_points, device='cuda') / num_points
    return p.multinomial(num_samples=num_samples, replacement=False)

def gen_idx_perm_cpu(num_points, num_samples):
    return torch.randperm(num_points)[:num_samples].to('cuda')

def gen_idx_perm_gpu(num_points, num_samples):
    return torch.randperm(num_points, device='cuda')[:num_samples]

def gen_idx_perm_gpu_f32(num_points, num_samples):
    return torch.randperm(num_points, dtype=torch.int32, device='cuda')[:num_samples]

idx_fns = [gen_idx_rand, 
    gen_idx_mult_cpu, 
    gen_idx_mult_gpu, 
    gen_idx_perm_cpu, 
    gen_idx_perm_gpu,
    gen_idx_perm_gpu_f32]

num_points = np.logspace(2, 7, 50, dtype=int)
num_samples = 50

d = []
for n_p in num_points:
    print(f"Testing functions with {n_p} points.")
    for fn in idx_fns:
        tic = time.time()
        samples = fn(n_p, num_samples)
        toc = time.time()
        assert type(samples) == torch.Tensor

        d.append({
            'Population': n_p,
            'Samples': num_samples,
            'Function':fn.__name__,
            'Time': toc - tic
        })

df = pd.DataFrame(d)

fig, ax = plt.subplots()

for fn in idx_fns:
    idxs = df['Function'] == fn.__name__
    ax.plot(df['Population'][idxs], df['Time'][idxs], label=fn.__name__)

ax.set_xlabel('Population Size')
ax.set_ylabel('Time')
ax.set_yscale('log')
ax.legend()
plt.savefig("figs/randperm.png")