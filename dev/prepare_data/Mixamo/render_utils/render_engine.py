"""
Blender render-engine configuration helpers.

Call ``configure_render_engine`` once after you have set up the scene
(resolution, output format, etc.) but before your frame loop.
"""

import bpy


# GPU backend preference order – first one that activates successfully wins.
_GPU_BACKENDS = ["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"]


def _try_enable_gpu_cycles(scene) -> bool:
    """
    Attempt to enable GPU rendering for Cycles.
    Returns True if at least one GPU device was activated, False otherwise.
    """
    try:
        scene.cycles.device = "GPU"
        prefs = bpy.context.preferences.addons["cycles"].preferences

        for backend in _GPU_BACKENDS:
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                n_enabled = 0
                for dev in prefs.devices:
                    if dev.type in _GPU_BACKENDS:
                        dev.use = True
                        n_enabled += 1
                if n_enabled > 0:
                    print(
                        f"[#] Cycles GPU: {backend} ({n_enabled} device(s) enabled)",
                        flush=True,
                    )
                    return True
            except Exception:
                continue

        print("[#] No compatible GPU device found, falling back to CPU.", flush=True)
        scene.cycles.device = "CPU"
        return False

    except Exception as exc:
        print(f"[#] GPU configuration error: {exc}. Using CPU.", flush=True)
        scene.cycles.device = "CPU"
        return False


def configure_render_engine(
    scene,
    engine: str = "cycles",
    samples: int = 16,
    use_gpu: bool = False,
    legacy_mode: bool = False,
):
    """
    Configure ``scene.render.engine``, samples, and device.

    Parameters
    ----------
    scene:
        The Blender scene to configure (``bpy.context.scene``).
    engine:
        ``'cycles'`` or ``'eevee'``.
    samples:
        Sample count (Cycles only).
    use_gpu:
        Attempt GPU rendering.  Silently falls back to CPU on failure.
    legacy_mode:
        If True, leave all render settings untouched (useful for debugging
        or comparing against default Blender behaviour).
    """
    if legacy_mode:
        print("[#] Legacy mode: render settings unchanged.", flush=True)
        return

    if engine.lower() == "cycles":
        print(
            f"[#] Cycles engine · {samples} samples · "
            f"{'GPU' if use_gpu else 'CPU'}",
            flush=True,
        )
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples

        if use_gpu:
            _try_enable_gpu_cycles(scene)
        else:
            scene.cycles.device = "CPU"
            print("[#] CPU rendering.", flush=True)

    else:  # eevee (default for older Blender, EEVEE-Next in 4.x)
        scene.render.engine = "BLENDER_EEVEE"
        print("[#] EEVEE engine.", flush=True)

    # General quality / speed tweaks
    scene.render.use_simplify = True
    scene.render.simplify_subdivision = 0
