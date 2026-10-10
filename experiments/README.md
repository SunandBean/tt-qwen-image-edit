# Verification scripts

These are the scripts behind the numbers in [`../PORTING_NOTES.md`](../PORTING_NOTES.md).

| Script | What it checks | Runs standalone |
|---|---|---|
| `check_vs_comfy.py` | Every stage against the ComfyUI modules on CPU in float32: tokenizer ids, template end, mrope positions, LM, vision tower, one DiT block. Results in `check_vs_comfy.json`. | needs ComfyUI importable |
| `device_check.py` | The same stages on the card: DiT 2 blocks, DiT 60 blocks first step, full text encoder, end to end. `STAGES=dit2,te2,te,dit,e2e` selects. | needs a p100a |
| `comfy_parity.py` | Runs the same prompt, size and seed through ComfyUI and through this port, then reports PSNR and PCC. | needs a p100a and a ComfyUI server |
| `vae_check.py` | The single-frame `conv3d` → `conv2d` substitution (expects PCC 1.0). | CPU |
| `te_bf16_cpu.py` | CPU bf16 text encoder against float32 — the 0.9949 baseline the device result is measured against. | CPU |
| `stage_service.py` | End-to-end timings through the HTTP model service, including the contract errors (422). | **no** — bound to the private service harness |
| `run_device.sh` | Container runner used for `device_check.py`: caps container memory. | needs the model image (`IMAGE`) |



## Running these outside the tree they were written in

These are the scripts as they were run, inside the private working tree this port was developed in.
They are published as the record behind the numbers on the model card, and most of them need two
edits before they will run from a clone of this repo:

1. **The package name.** Four of them (`check_vs_comfy.py`, `device_check.py`, `te_bf16_cpu.py`,
   `vae_check.py`) do `from qwenedit import ...`. `qwenedit` is this port, published here as
   **`tt_qwen_image_edit`**; the module layout is the same.
2. **The `sys.path` line.** The same four insert `ROOT.parents[1] / "deploy"` (or `$DEPLOY`), the
   private tree's package directory. From a clone that is `ROOT.parent`.

`check_vs_comfy.py` also needs a ComfyUI checkout on `PYTHONPATH`. `comfy_parity.py` runs as published
against a ComfyUI server. `stage_service.py` is included because it documents how the published
end-to-end timings were produced: it drives the private deployment's own service (`tt_service.py`, its
`PHOTO_*` settings, its card hand-off), none of which is published here, so it is a record rather than
something to run. The device scripts need a p100a.
