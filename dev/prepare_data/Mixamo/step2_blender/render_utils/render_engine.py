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
        prefs.refresh_devices()
        print("[#] Available compute devices:")
        for d in prefs.devices:
            print(d.name, d.type, d.use)

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
    enable_gi: bool = False,
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
    enable_gi:
        If True, enable global illumination (more expensive).
    """
    print("#" * 100, flush=True)
    if legacy_mode:
        print("[#] Legacy mode: render settings unchanged.", flush=True)
        return

    if engine.lower() == "cycles":
        print("[#] Configuring Cycles render engine...", flush=True)
        print(
            f"[#] Cycles engine · {samples} samples · "
            f"{'GPU' if use_gpu else 'CPU'}",
            flush=True,
        )
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        if not enable_gi:
            print("[#] Disabling global illumination for faster renders (may cause darker shadows and less realistic lighting).", flush=True)
            scene.cycles.max_bounces          = 0   # total
            scene.cycles.diffuse_bounces      = 0   # no colour bleeding / GI
            scene.cycles.glossy_bounces       = 0   # keep one for basic reflections
            scene.cycles.transmission_bounces = 0
            scene.cycles.volume_bounces       = 0
            scene.cycles.transparent_max_bounces = 0
            # ── Clamping — crush any residual bright indirect samples ────────────────────
            scene.cycles.sample_clamp_indirect = 0   # 0 = off, lower = more suppression
            scene.cycles.sample_clamp_direct   = 0   # leave direct light unclamped
            # ── Caustics — off ───────────────────────────────────────────────────────────
            scene.cycles.caustics_reflective = False
            scene.cycles.caustics_refractive = False
            # Tile size: larger tiles are usually better for GPU, smaller for CPU (but this is scene-dependent)

        if use_gpu:
            _try_enable_gpu_cycles(scene)
            print(f"[#] Successfully enabled GPU rendering for Cycles (Status = {bpy.context.scene.cycles.device}).", flush=True)
        else:
            scene.cycles.device = "CPU"
            print("[#] CPU rendering.", flush=True)

    else:  # eevee (default for older Blender, EEVEE-Next in 4.x)
        scene.render.engine = "BLENDER_EEVEE"
        print("[#] EEVEE engine.", flush=True)
        
    # scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_use_gpu = use_gpu
    # scene.cycles.use_denoising = False  # for now, to isolate render quality variables
    print("Device:", scene.cycles.device)
    print("Samples:", scene.cycles.samples)
    print("Denoising:", scene.cycles.use_denoising)
    print("Denoiser:", scene.cycles.denoiser)
    # General quality / speed tweaks
    scene.render.use_simplify = True
    scene.render.simplify_subdivision = 0
    print("#" * 100, flush=True)
