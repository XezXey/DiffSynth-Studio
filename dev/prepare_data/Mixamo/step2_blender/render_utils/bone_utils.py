"""
Utilities for locating bones in Mixamo armatures.

Mixamo uses several naming conventions:
  - mixamorig:Hips
  - mixamorig1:Hips
  - mixamorig2:Hips
  - Hips  (no prefix)

Use find_bone_by_suffix(armature, suffix) to locate a bone regardless of
which convention the file uses.
"""


def find_bone_by_suffix(armature, suffix: str):
    """
    Return the full pose-bone name whose short name matches *suffix*.

    Checks exact match, colon-prefixed match, then falls back to a
    case-insensitive search.  Returns ``None`` if nothing is found.
    """
    for bone in armature.pose.bones:
        if bone.name == suffix or bone.name.endswith(":" + suffix):
            return bone.name

    suffix_lower = suffix.lower()
    for bone in armature.pose.bones:
        name_lower = bone.name.lower()
        if name_lower == suffix_lower or name_lower.endswith(":" + suffix_lower):
            return bone.name

    return None


def resolve_follow_bone(arm, requested_name: str) -> str:
    """
    Given an armature object and a requested bone name (which may or may not
    exist verbatim), return the actual bone name to use.

    Raises ``RuntimeError`` if no matching bone can be found.
    """
    existing = [pb.name for pb in arm.pose.bones]

    if requested_name in existing:
        return requested_name

    suffix = requested_name.split(":")[-1]
    detected = find_bone_by_suffix(arm, suffix)
    if detected:
        print(
            f"[#] Auto-detected follow bone: {detected} "
            f"(requested: {requested_name})"
        )
        return detected

    available = ", ".join(existing[:10])
    raise RuntimeError(
        f"Could not find bone '{requested_name}' or suffix '{suffix}'. "
        f"Available bones (first 10): {available} ..."
    )
