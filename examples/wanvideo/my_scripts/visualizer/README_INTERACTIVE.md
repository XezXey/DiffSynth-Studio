# Interactive Multi-Model Skeleton Comparison

🎉 **New Features Added!**

## What's New

The visualizer now supports interactive skeleton visibility toggling, perfect for comparing multiple models performing the same motion.

### Interactive Controls

1. **Click badges** to hide/show individual skeletons
2. **Shift+Click** a badge to show ONLY that skeleton (solo mode)  
3. **Press 1-9** keys to toggle skeletons 1-9 across all panels
4. **Visual feedback**: Hidden skeletons shown with strikethrough + reduced opacity

## Usage Examples

### Basic: Comparing 4 Models

```python
from viz3d_utils import generate_html, panel_3d

# Your model predictions (each is T×J×3)
model_a_motion = ...  # shape: (80, 22, 3)
model_b_motion = ...
model_c_motion = ...
ground_truth   = ...

# Create a 3D panel with all skeletons
rows = [{
    "label": "Model Comparison",
    "T": 80,
    "panels": [
        panel_3d(skeletons=[
            {"joints": model_a_motion, "color": "#4ECDC4", "label": "Model A"},
            {"joints": model_b_motion, "color": "#FF6B6B", "label": "Model B"},
            {"joints": model_c_motion, "color": "#FFB86C", "label": "Model C"},
            {"joints": ground_truth,   "color": "#50FA7B", "label": "GT"},
        ])
    ],
}]

generate_html(rows, output_path="comparison.html")
```

### Advanced: Multi-Panel Layout

```python
# Show same skeletons in 3D, 2D, and image overlay
rows = [{
    "label": "Full Comparison",
    "T": 80,
    "panels": [
        # 3D view with all models
        panel_3d(skeletons=[
            {"joints": model_a_motion, "color": "#4ECDC4", "label": "Model A"},
            {"joints": model_b_motion, "color": "#FF6B6B", "label": "Model B"},
            {"joints": ground_truth,   "color": "#50FA7B", "label": "GT"},
        ], label="3D View"),
        
        # 2D projection
        panel_2d(skeletons=[
            {"joints": model_a_motion[:,:,:2], "color": "#4ECDC4", "label": "Model A"},
            {"joints": model_b_motion[:,:,:2], "color": "#FF6B6B", "label": "Model B"},
            {"joints": ground_truth[:,:,:2],   "color": "#50FA7B", "label": "GT"},
        ], label="2D View"),
        
        # Video overlay (pred vs GT)
        panel_image_overlay(
            images=video_frames,
            joints=model_a_motion[:,:,:2],
            joints_gt=ground_truth[:,:,:2],
            joint_color="#4ECDC4",
            gt_color="#50FA7B",
            label="Video Overlay"
        ),
    ],
}]
```

### Legacy API Still Works

The old API is still fully supported:

```python
panel_3d(
    pred=model_a_motion,
    gt=ground_truth,
    extra_skeletons=[
        {"joints": model_b_motion, "color": "#FF6B6B", "label": "Model B"},
        {"joints": model_c_motion, "color": "#FFB86C", "label": "Model C"},
    ]
)
```

## Workflow Tips

### Comparing 4+ Models

When you have many models, the interactive toggles become essential:

1. **Start with all visible** - Get overall sense of differences
2. **Use solo mode** (Shift+Click) - Inspect each model individually  
3. **Pairwise comparison** - Click to hide all but two models
4. **Keyboard shortcuts** - Quickly toggle specific models (Press 1, 2, 3, etc.)

### Example Workflow

```
Initial view: All 5 skeletons visible (cluttered)
  ↓
Shift+Click "Model A" → Solo mode (only Model A visible)
  ↓
Click "GT" badge → Add GT back (compare A vs GT)
  ↓  
Click "Model A" → Hide it, Click "Model B" → Add it (compare B vs GT)
  ↓
Press "1" → Toggle Model A back on/off quickly
```

## Color Palette Suggestions

For 4-6 models, use these distinct colors:

```python
COLORS = {
    "model_a": "#4ECDC4",  # Teal
    "model_b": "#FF6B6B",  # Red
    "model_c": "#FFB86C",  # Orange
    "model_d": "#BD93F9",  # Purple
    "model_e": "#F1FA8C",  # Yellow
    "gt":      "#50FA7B",  # Green
}
```

## Demo

Run the included demo:

```bash
python demo_multi_model.py
```

This generates `multi_model_demo.html` with 5 skeletons across multiple panels.

## Benefits

✅ **No page reload** - Toggle visibility instantly  
✅ **Works across all panel types** - 3D, 2D, and image overlay  
✅ **Synchronized** - All panels in a row share the same playback  
✅ **Keyboard shortcuts** - Fast comparison workflow  
✅ **Solo mode** - Focus on one model at a time  

---

Perfect for:
- Model ablation studies
- Hyperparameter comparison
- Architecture comparison
- Ensemble analysis
- Debugging motion artifacts
