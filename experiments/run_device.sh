#!/usr/bin/env bash
# Run device_check.py (or SCRIPT) in the model image on the P100a.
# Set IMAGE to the tt-metal / ttnn image the port was built against; there is no portable default.
# The container is memory-capped: on the host this was developed on, most RAM was held by a ComfyUI model cache
# (~35 GB); an uncapped run once pushed the host into a global OOM that killed ComfyUI instead of this container.
#   STAGES=dit2,te2 ./run_device.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEV="$(cd "$HERE/../.." && pwd)"
IMAGE="${IMAGE:?set IMAGE to the tt-metal image the port was built against}"
SCRIPT="${SCRIPT:-device_check.py}"
TOKENIZER=/hf/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
exec docker run --rm --name qwenedit-dev --ipc host --device /dev/tenstorrent \
    --memory "${MEMORY:-14g}" --memory-swap "${MEMORY:-14g}" \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    -v "$HOME/.cache/huggingface:/hf:ro" -v "$HOME/ComfyUI/models:/comfy:ro" \
    -v "$DEV/deploy:/service:ro" -v "$HERE:/work" \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TT_METAL_VISIBLE_DEVICES=0 -e MESH_DEVICE=P100 \
    -e QWEN_EDIT_MODELS=/comfy -e QWEN_EDIT_TOKENIZER="$TOKENIZER" -e DEPLOY=/service \
    -e STAGES="${STAGES:-dit2,te2,te,dit,e2e}" -e THREADS="${THREADS:-16}" -e SIZES="${SIZES:-1024x1024,1024x1024,832x1216}" \
    -e PROMPT="${PROMPT:-Change the season to winter with snow on the ground}" \
    -e WAN_SIZES -e EXPERT -e FFN_BFP4 -e PYTHONPATH=/service:/opt/tt-metal \
    --workdir /work --entrypoint python "$IMAGE" "/work/$SCRIPT"
