# SPDX-License-Identifier: Apache-2.0
"""Device helpers for a single Tenstorrent Blackhole p100a: open/close, DRAM stats, timing."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict




def open_device(trace_region_size: int = 0, l1_small_size: int = 98304):
    import ttnn

    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
        trace_region_size=trace_region_size,
        l1_small_size=l1_small_size,
    )


def close_device(dev):
    import ttnn

    ttnn.close_mesh_device(dev)


def dram_stats(dev) -> Dict[str, int]:
    import ttnn

    ttnn.synchronize_device(dev)
    v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    banks = int(v.num_banks)
    return {"total_bytes": int(v.total_bytes_per_bank) * banks,
            "allocated_bytes": int(v.total_bytes_allocated_per_bank) * banks,
            "free_bytes": int(v.total_bytes_free_per_bank) * banks,
            "largest_free_bytes_per_bank": int(v.largest_contiguous_bytes_free_per_bank)}


@dataclass
class Timing:
    values: Dict[str, float] = field(default_factory=dict)

    def mark(self, key: str, started: float):
        self.values[key] = self.values.get(key, 0.0) + time.perf_counter() - started
