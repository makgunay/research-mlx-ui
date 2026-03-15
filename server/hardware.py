"""Auto-detect Apple Silicon chip and memory. Returns specs + tuned recommendations."""

import re
import subprocess


def detect_hardware() -> dict:
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=10,
        )
        info = result.stdout
    except Exception:
        return _fallback_hardware()

    chip_match = re.search(r"Chip:\s+(.+)", info)
    memory_match = re.search(r"Memory:\s+(\d+)\s*GB", info)

    chip = chip_match.group(1).strip() if chip_match else "Apple Silicon"
    mem_gb = int(memory_match.group(1)) if memory_match else 16

    return {
        "chip": chip,
        "memory": f"{mem_gb}GB",
        "recommendations": _get_recommendations(chip, mem_gb),
    }


def _get_recommendations(chip: str, mem_gb: int) -> dict:
    """Conservative defaults for the detected hardware. The agent can push these."""
    if mem_gb >= 64:
        return {"depth": 4, "max_seq_len": 512, "device_batch_size": 32,
                "muon_lr": 0.02, "adamw_lr": 1e-3}
    elif mem_gb >= 32:
        return {"depth": 4, "max_seq_len": 512, "device_batch_size": 24,
                "muon_lr": 0.02, "adamw_lr": 1e-3}
    elif mem_gb >= 16:
        return {"depth": 4, "max_seq_len": 512, "device_batch_size": 16,
                "muon_lr": 0.02, "adamw_lr": 1e-3}
    else:
        return {"depth": 3, "max_seq_len": 256, "device_batch_size": 8,
                "muon_lr": 0.015, "adamw_lr": 5e-4}


def _fallback_hardware() -> dict:
    return {
        "chip": "Apple Silicon (detection failed)",
        "memory": "Unknown",
        "recommendations": {"depth": 4, "max_seq_len": 256, "device_batch_size": 8,
                            "muon_lr": 0.02, "adamw_lr": 3e-4},
    }
