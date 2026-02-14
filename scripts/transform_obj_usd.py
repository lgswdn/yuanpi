#!/usr/bin/env python

"""
Converts a directory of OBJ/GLB mesh files into normalized USD files for Isaac Lab.

This script performs the following steps for each mesh:
1.  Loads the mesh (OBJ or GLB).
2.  Centers the mesh at its centroid.
3.  Applies a +90 degree rotation around the X-axis to convert from Y-up to Z-up.
4.  Normalizes the mesh to have a unit diameter (max distance from origin = 0.5).
5.  Saves this normalized mesh to a dedicated directory.
6.  Calls Isaac Lab's 'convert_mesh.py' to create a USD.
7.  Calls Isaac Lab's 'post_process_usd.py' to add rigid body properties.
"""

import os
import sys
import numpy as np

# --- Check for dependencies ---
try:
    import trimesh
except ImportError:
    print("[ERROR] 'trimesh' library not found.")
    print("Please install it: pip install trimesh")
    sys.exit(1)

# --- Configuration ---

# 1. Define paths
# Use os.path.expanduser to correctly resolve the '~' (home) directory
in_dir = os.path.expanduser("~/DGN/coacd")
out_dir = os.path.expanduser("~/DGN/coacd/usds")
normalized_dir = os.path.expanduser("~/DGN/coacd/usd_normalized")

# 2. Define path to Isaac Lab root to find tool scripts
isaaclab_root = os.path.expanduser("~/IsaacLab")
convert_script_path = os.path.join(isaaclab_root, "scripts/tools/convert_mesh.py")
post_process_script_path = os.path.join(isaaclab_root, "scripts/tools/post_process_usd.py")

# 3. Define physics properties
# These were hardcoded in your original script
MESH_MASS = 0.5
COLLISION_PRESET = "convexHull"

# --- End Configuration ---


def normalize_obj_to_unit_diameter(obj_path, normalized_obj_path):
    """Normalize OBJ/GLB file to have diameter of 1.0."""
    
    # Load mesh
    try:
        mesh = trimesh.load(obj_path, force='mesh')
    except Exception as e:
        print(f'[ERROR] Failed to load mesh {obj_path}: {e}')
        return False

    if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
        print(f'[ERROR] No vertices found in {obj_path}')
        return False

    # Center the mesh at origin first
    try:
        mesh.apply_translation(-mesh.centroid)
    except Exception as e:
        print(f'[WARNING] Could not center mesh {obj_path}. It might be empty or invalid. Error: {e}')
        # Continue anyway, scaling might still be possible
        pass

    # Convert from Y-up to Z-up coordinate system for IsaacSim/USD
    # Rotate +90 degrees around X-axis: Y-up -> Z-up
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=np.pi/2,  # +90 degrees in radians
        direction=[1, 0, 0],  # X-axis
        point=[0, 0, 0]  # Origin
    )
    mesh.apply_transform(rotation_matrix)
    print(f'[TRANSFORM] Applied Y-up to Z-up conversion (rotation +90° around X-axis)')

    # Calculate the maximum distance from any vertex to origin
    distances = np.linalg.norm(mesh.vertices, axis=1)
    max_distance = np.max(distances)

    # Use a small epsilon to avoid division by zero for single-point meshes
    if max_distance > 1e-6:
        # Calculate scale factor so that max distance becomes 0.5
        # This makes the diameter (max distance * 2) equal to 1.0
        scale_factor = 0.5 / max_distance

        # Scale the mesh
        mesh.apply_scale(scale_factor)
        
        print_scale = f'scale: {scale_factor:.4f}'
    else:
        # Mesh is likely a single point or has no extent
        print_scale = 'scale: 1.0 (mesh has no extent)'
        
    # Export normalized mesh
    try:
        mesh.export(normalized_obj_path)
        print(f'[NORMALIZE] {os.path.basename(obj_path)} -> max_distance: {max_distance:.4f} -> {print_scale}')
        return True
    except Exception as e:
        print(f'[ERROR] Failed to export normalized mesh {normalized_obj_path}: {e}')
        return False


def main():
    """Main execution function."""
    
    # Create normalized directory if it doesn't exist
    os.makedirs(normalized_dir, exist_ok=True)
    print(f'[INFO] Input directory:      {in_dir}')
    print(f'[INFO] Output directory:     {out_dir}')
    print(f'[INFO] Normalized files dir: {normalized_dir}')

    # Scan all OBJ and GLB files in the input directory
    mesh_files = []
    try:
        for filename in os.listdir(in_dir):
            base_name, ext = os.path.splitext(filename)
            if ext.lower() in ['.obj', '.glb']:
                mesh_files.append((base_name, ext))
    except FileNotFoundError:
        print(f'[ERROR] Input directory not found: {in_dir}')
        sys.exit(1)
    except Exception as e:
        print(f'[ERROR] Could not scan directory {in_dir}: {e}')
        sys.exit(1)


    print(f'[INFO] Found {len(mesh_files)} mesh files (OBJ/GLB) to process')
    if not mesh_files:
        print('[INFO] No files to process. Exiting.')
        return

    count = 0
    for base, ext in mesh_files:
        count += 1
        print(f'\n--- Processing item {count}/{len(mesh_files)}: {base}{ext} ---')

        # Define all paths for this file
        obj_path = os.path.join(in_dir, f'{base}{ext}')
        
        # This matches your original structure: /usds/[basename]/_[basename].usd
        out_path_for_asset = os.path.join(out_dir, base)
        os.makedirs(out_path_for_asset, exist_ok=True)
        usd_path = os.path.join(out_path_for_asset, f'_{base}.usd')
        
        # This preserves the original extension (e.g., .obj or .glb)
        normalized_mesh_path = os.path.join(normalized_dir, f'{base}{ext}')

        if not os.path.exists(obj_path):
            print(f'[ERROR] Input file not found: {obj_path}')
            continue

        if os.path.exists(usd_path):
            print(f'[SKIP] Output file {usd_path} already exists')
            continue

        # --- Step 1: Normalization ---
        if os.path.exists(normalized_mesh_path):
            print(f'[SKIP] Normalized file already exists: {normalized_mesh_path}')
        else:
            # Normalize input file to unit diameter
            print(f'[NORMALIZE] Processing {obj_path} -> {normalized_mesh_path}')
            if not normalize_obj_to_unit_diameter(obj_path, normalized_mesh_path):
                print(f'[SKIP] Failed to normalize {obj_path}')
                continue
        
        # --- Step 2: Convert to USD ---
        print(f'[CONVERT] {normalized_mesh_path} -> {usd_path}')
        
        # Note: The '"{}"' quotes are important for the shell to handle paths with spaces
        cmd_convert = (
            f'python "{convert_script_path}" '
            f'"{normalized_mesh_path}" "{usd_path}" '
            f'--mass {MESH_MASS} '
            f'--collision-approximation {COLLISION_PRESET} '
            '--make-instanceable --headless'
        )
        
        print(f'[RUNNING] {cmd_convert}')
        return_code = os.system(cmd_convert)

        if return_code != 0:
            print(f'[ERROR] USD conversion failed for {normalized_mesh_path} with exit code: {return_code}')
            continue

        # --- Step 3: Post-process USD ---
        if os.path.exists(usd_path):
            print(f'[POST-PROCESS] Adding rigid body properties to {usd_path}')
            
            cmd_post_process = (
                f'python "{post_process_script_path}" '
                f'"{usd_path}" '
                f'--mass {MESH_MASS} '
                f'--colliders-preset {COLLISION_PRESET} --headless'
            )

            print(f'[RUNNING] {cmd_post_process}')
            return_code_post = os.system(cmd_post_process)

            if return_code_post != 0:
                print(f'[ERROR] USD post-processing failed for {usd_path} with exit code: {return_code_post}')
            else:
                print(f'[SUCCESS] Successfully processed {base}{ext} -> {usd_path}')
        else:
            print(f'[ERROR] Failed to create {usd_path} during conversion step.')

        if os.path.exists(normalized_mesh_path):
            print(f'[INFO] Normalized file saved: {normalized_mesh_path}')

    print("\n[INFO] All files processed.")

if __name__ == "__main__":
    main()