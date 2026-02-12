import torch as th
import numpy as np
import os
from importlib import reload
import matplotlib.pyplot as plt
import vae_decoder
import tqdm
reload(vae_decoder)

def require_grads(mod):
    print("# Check Required grads on given modules...")
    for n, p in mod.named_parameters():
        print(n, p.shape, p.requires_grad)
    print("#" * 100)

# Dev
device = 'cuda'
J = 25
out_J_chn = 2
vae_decoder = vae_decoder.JointVAE(J=J, out_J_chn=out_J_chn)
vae_decoder = vae_decoder.to(device)
require_grads(vae_decoder)
vae_decoder.train()
for n, p in vae_decoder.named_parameters():
    p.requires_grad = True
require_grads(vae_decoder)
optimizer = th.optim.Adam(params=vae_decoder.parameters(), lr=1e-5)

# gt = th.zeros((1, 3, 61, 64, 64)).to(device)
gt = th.randn((1, 3, 61, 64, 64)).to(device)
x = th.randn(1, 16, 16, 8, 8).to(device)
loss_list = []
for i in tqdm.tqdm(range(300), desc="Iterations: "):
    optimizer.zero_grad()
    out = vae_decoder.decode(x, device=device)
    print(out.shape)
    loss = th.mean((gt - out)**2)
    print(loss)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())
    
plt.plot(list(range(len(loss_list))), loss_list, '-x')
plt.savefig('loss.png')