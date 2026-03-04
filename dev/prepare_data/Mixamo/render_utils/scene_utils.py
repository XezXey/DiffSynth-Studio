"""
Blender scene-management helpers.

Functions here should be called once per FBX file (or once per scene setup).
They intentionally have no return values so they can be used as simple
"configure the scene" commands.
"""

import bpy


# ---------------------------------------------------------------------------
# Scene lifecycle
# ---------------------------------------------------------------------------

def clear_scene():
    """Delete every object in the current scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    print("[#] Scene cleared.")


def load_fbx(filepath: str, char_color=None):
    """
    Import an FBX file into the current scene.

    Parameters
    ----------
    filepath:
        Absolute path to the ``.fbx`` file.
    char_color:
        If provided, override every mesh material with a solid black Principled
        BSDF.  Pass any truthy value to enable; the colour is always black so
        that silhouette / skeleton overlays stay clean.
    """
    bpy.ops.import_scene.fbx(filepath=filepath)
    print(f"[#] Loaded FBX: {filepath}")

    if char_color is not None:
        print(f"[#] Applying flat character colour override")
        mat = bpy.data.materials.new(name="FlatColor")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bsdf.inputs["Roughness"].default_value = 1.0

        for obj in bpy.context.selected_objects:
            if obj.type == "MESH":
                obj.data.materials.clear()
                obj.data.materials.append(mat)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def create_default_camera():
    """
    Add a plain camera at (0, -3, 1.5) pointing toward the origin and set it
    as the scene camera.  Returns the camera object.
    """
    bpy.ops.object.camera_add(
        location=(0, -3, 1.5),
        rotation=(1.5708, 0, 0),
    )
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    print("[#] Default camera created.")
    return cam


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

def ensure_sun_light():
    """
    Guarantee that at least one Sun light exists in the scene.
    Returns the (possibly newly-created) Sun light object.
    """
    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and obj.data.type == "SUN":
            return obj

    light_data = bpy.data.lights.new(name="Sun", type="SUN")
    light_data.energy = 3.0

    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = (10, -10, 10)
    light_obj.rotation_euler = (0.7, 0.0, 0.8)

    print("[#] Sun light created.")
    return light_obj


# ---------------------------------------------------------------------------
# Background / World
# ---------------------------------------------------------------------------

def setup_white_background():
    """Set the world background to solid white."""
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1, 1, 1, 1)
    bg.inputs[1].default_value = 1.0
