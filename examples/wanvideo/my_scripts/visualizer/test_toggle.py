#!/usr/bin/env python3
"""Quick test for toggle functionality - creates 4 skeletons to test hiding."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Check if numpy/PIL are available
try:
    from viz3d_utils import generate_html, panel_3d, _demo_skeleton
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying without numpy...")
    # Create minimal test without dependencies
    html = """<!DOCTYPE html>
<html><body><h1>Please install numpy and Pillow to run test</h1>
<pre>pip install numpy pillow</pre></body></html>"""
    Path("test_toggle.html").write_text(html)
    print("Created minimal test_toggle.html")
    sys.exit(0)

print("Creating test with 4 distinct skeletons...")

T = 40
colors = ["#4ECDC4", "#FF6B6B", "#FFB86C", "#BD93F9"]
labels = ["Skeleton 1", "Skeleton 2", "Skeleton 3", "Skeleton 4"]

# Create 4 different skeleton motions
skeletons = []
for i, (color, label) in enumerate(zip(colors, labels)):
    # Each skeleton has slightly different motion
    skel = _demo_skeleton(T=T, n_joints=22, seed=i*10)
    # Offset them spatially so they don't overlap completely
    offset = np.array([i * 0.3 - 0.45, 0, 0])
    skel = skel + offset[None, None, :]
    
    skeletons.append({
        "joints": skel,
        "color": color,
        "label": label
    })

print(f"  ✓ Created {len(skeletons)} skeletons")

rows = [{
    "label": "Toggle Test - Click badges or press 1,2,3,4 keys",
    "T": T,
    "panels": [
        panel_3d(skeletons=skeletons, label="All 4 Skeletons"),
    ],
}]

output_path = Path(__file__).parent / "test_toggle.html"
generate_html(
    rows=rows,
    output_path=str(output_path),
    title="Toggle Test - 4 Skeletons",
    cell_height=600,
    joint_set="smpl22"
)

size_kb = output_path.stat().st_size / 1024
print(f"\n✅ Test file created: {output_path.name} ({size_kb:.1f} KB)")
print(f"\n📋 Test instructions:")
print(f"   1. Open {output_path.name} in browser")
print(f"   2. Click 'Skeleton 1' badge → Should hide skeleton 1 ONLY")
print(f"   3. Click again → Should show skeleton 1 again")  
print(f"   4. Press '1' key → Should toggle skeleton 1")
print(f"   5. Press '2' key → Should toggle skeleton 2")
print(f"   6. Shift+Click 'Skeleton 3' → Should show ONLY skeleton 3")
print(f"   7. Verify no wrong skeletons hide/show")
