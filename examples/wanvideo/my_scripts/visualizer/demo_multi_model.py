#!/usr/bin/env python3
"""
Demo: Multi-Model Comparison with Interactive Skeleton Toggling
================================================================

This demo shows how to compare multiple models performing the same motion
using the new interactive features:
  • Click badges to hide/show individual models
  • Shift+Click to show only one model (solo mode)
  • Press 1-9 keys to toggle models across all panels
"""

import numpy as np
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))
from viz3d_utils import generate_html, panel_3d, panel_2d, panel_image_overlay, _demo_skeleton, _demo_images


def create_model_variations(base_motion, n_models=4, noise_scale=0.05, seed=42):
    """Create variations of a base motion to simulate different model predictions."""
    rng = np.random.default_rng(seed)
    T, J, D = base_motion.shape
    
    models = []
    colors = ["#4ECDC4", "#FF6B6B", "#FFB86C", "#BD93F9", "#50FA7B", "#F1FA8C"]
    
    for i in range(n_models):
        # Add different types of noise to simulate model differences
        noise = rng.standard_normal((T, J, D)) * noise_scale
        
        # Add temporal drift (some models drift over time)
        drift = np.linspace(0, 0.1 * i, T)[:, None, None] * rng.standard_normal((1, J, D))
        
        # Add spatial bias (some models have systematic offset)
        bias = rng.standard_normal((1, J, D)) * 0.02 * i
        
        variation = base_motion + noise + drift + bias
        
        models.append({
            "joints": variation,
            "color": colors[i % len(colors)],
            "label": f"Model {chr(65+i)}"  # A, B, C, D...
        })
    
    return models


def main():
    print("🎨 Generating multi-model comparison demo...")
    
    # Generate base motion (ground truth)
    T = 80
    gt_motion = _demo_skeleton(T=T, n_joints=22, seed=100)
    
    # Create 4 model variations
    models = create_model_variations(gt_motion, n_models=4, noise_scale=0.08)
    
    # Add ground truth as the last skeleton
    models.append({
        "joints": gt_motion,
        "color": "#50FA7B",
        "label": "Ground Truth"
    })
    
    print(f"  ✓ Created {len(models)} skeleton variations")
    
    # Generate some demo images
    images = _demo_images(T=T, W=320, H=240)
    print(f"  ✓ Generated {len(images)} demo frames")
    
    # Create visualization with multiple panels
    rows = [
        {
            "label": "Multi-Model Comparison — 3D View",
            "T": T,
            "panels": [
                panel_3d(
                    skeletons=models,
                    label="3D — All Models"
                ),
            ],
        },
        {
            "label": "Multi-Model Comparison — 3D + 2D + Overlay",
            "T": T,
            "panels": [
                panel_3d(
                    skeletons=models,
                    label="3D View"
                ),
                panel_2d(
                    skeletons=[{**m, "joints": m["joints"][:, :, :2]} for m in models],
                    label="2D View"
                ),
                panel_image_overlay(
                    images=images,
                    # For overlay, just show first two models to avoid clutter
                    joints=models[0]["joints"][:, :, :2],
                    joints_gt=models[-1]["joints"][:, :, :2],  # GT
                    joint_color=models[0]["color"],
                    gt_color=models[-1]["color"],
                    label="Image Overlay"
                ),
            ],
        },
        {
            "label": "Model A vs Model B vs Ground Truth",
            "T": T,
            "panels": [
                panel_3d(
                    skeletons=[models[0], models[1], models[-1]],
                    label="Compare A, B, GT"
                ),
            ],
        },
    ]
    
    output_path = Path(__file__).parent / "multi_model_demo.html"
    generate_html(
        rows=rows,
        output_path=str(output_path),
        title="Multi-Model Skeleton Comparison",
        cell_height=720,
        joint_set="smpl22"
    )
    
    file_size = output_path.stat().st_size / 1024
    print(f"\n✅ Demo created: {output_path}")
    print(f"   File size: {file_size:.1f} KB")
    print(f"\n💡 Interactive features:")
    print(f"   • Click skeleton badges to hide/show models")
    print(f"   • Shift+Click a badge to show ONLY that model")
    print(f"   • Press 1-5 keys to toggle models 1-5")
    print(f"\n🌐 Open {output_path.name} in your browser to explore!")


if __name__ == "__main__":
    main()
