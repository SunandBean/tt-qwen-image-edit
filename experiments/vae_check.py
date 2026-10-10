"""CPU: single-frame 2D path of the VAE vs diffusers' conv3d path (small size), then time + peak memory at 1024^2."""
import os, sys, time, resource, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("DEPLOY") or str(ROOT.parents[1] / "deploy"))
from qwenedit.vae import QwenImageVAE
from qwenedit.config import paths
torch.set_num_threads(int(os.environ.get("THREADS", "16")))
def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double(); a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))
def peak():
    for f in ("/sys/fs/cgroup/memory.peak",):
        try: return round(int(Path(f).read_text()) / 2**30, 2)
        except OSError: pass
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
fast = QwenImageVAE(paths()["vae"], dtype=torch.float32)
slow = QwenImageVAE(paths()["vae"], dtype=torch.float32)
for m in slow.vae.modules():
    if "forward" in m.__dict__: del m.forward
torch.manual_seed(0)
img = torch.rand(1, 192, 256, 3)
with torch.no_grad():
    za, zb = fast.encode(img), slow.encode(img)
    lat = torch.randn(1, 16, 1, 24, 32)
    from qwenedit import host
    ya = fast.vae.decode(host.latent_out(lat)).sample; yb = slow.vae.decode(host.latent_out(lat)).sample
print({"enc_pcc": pcc(za, zb), "enc_maxdiff": float((za - zb).abs().max()), "dec_pcc": pcc(ya, yb), "dec_maxdiff": float((ya - yb).abs().max())})
del slow
v = QwenImageVAE(paths()["vae"])
print("peak before", peak())
t0 = time.time(); z = v.encode(torch.rand(1, 1024, 1024, 3)); t1 = time.time()
print("enc 1024", round(t1 - t0, 2), "peak", peak())
t0 = time.time(); im = v.decode(host.latent_in(z)); t1 = time.time()
print("dec 1024", round(t1 - t0, 2), "peak", peak(), im.size)
