# import numpy as np
import json
import os
import math
import argparse
import collections
import re
import bpy

if __name__ == "__main__":
    import sys
    argv = sys.argv

    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--")+1:]

    print("argsv:{0}".format(argv))
    parser = argparse.ArgumentParser(description='Render Motion in 3D Environment.')
    parser.add_argument('--folder', type=str, metavar='PATH',
                        help='path to specific folder which include folders containing .obj files',
                        default='')
    parser.add_argument('--condition-folder', type=str, metavar='PATH',
                        help='path to specific folder which include input condityions containing .ply files',
                        default='')
    parser.add_argument('--out-folder', type=str, metavar='PATH',
                        help='path to output folder which include rendered img files',
                        default='')
    parser.add_argument('--scene', type=str, metavar='PATH',
                        help='path to specific .blend path for 3D scene',
                        default='')
    parser.add_argument('--reset-camera', action='store_true',
                        help='reset camera to default position',
                        default=False)
    args = parser.parse_args(argv)
    print("args:{0}".format(args))

    # Load the world
    WORLD_FILE = args.scene
    bpy.ops.wm.open_mainfile(filepath=WORLD_FILE)

    # Render Optimizations
    bpy.context.scene.render.use_persistent_data = True

    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

    scene_name = args.scene.split("/")[-1].replace("_scene.blend", "")
    print("scene name:{0}".format(scene_name))
    
    if args.reset_camera:
        if 'floor_colorful_mat.blend' in WORLD_FILE:
            camera = bpy.data.objects.get("Camera")
            if camera is not None:
                location = camera.location[:]
                rotation_euler = camera.rotation_euler[:]
                
                camera.location = (-location[0], location[1], location[2])
                camera.rotation_euler = (rotation_euler[0], rotation_euler[1], -rotation_euler[2])

                bpy.context.scene.camera = bpy.data.objects["Camera"]
            else:
                print("Camera not found.")
        else:
            camera = bpy.data.objects.get("Camera")
            if camera is not None:
                location = camera.location[:]
                rotation_euler = camera.rotation_euler[:]
                print(location, rotation_euler)
                camera.location = (location[0], location[1], location[2])
                camera.rotation_euler = (rotation_euler[0], rotation_euler[1], rotation_euler[2]+3.14)

                bpy.context.scene.camera = bpy.data.objects["Camera"]
            else:
                print("Camera not found.")

    obj_folder = args.folder
    condition_folder = args.condition_folder 
    output_dir = args.out_folder
    print("obj_folder:{0}".format(obj_folder))
    print("output dir:{0}".format(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load all the objs of input condition. 
    # Iterate folder to process all model
    if not os.path.exists(condition_folder):
        print(f"Condition folder {condition_folder} does not exist. Skipping condition loading.")
        condition_files = []
    else:
        condition_files = os.listdir(condition_folder)
    for c_name in condition_files:
        condition_path_to_file = os.path.join(condition_folder, c_name)

        # Load human mesh and set material 
        if ".obj" in condition_path_to_file:
            condition_new_obj = bpy.ops.import_scene.obj(filepath=condition_path_to_file, split_mode ="OFF")
        elif ".ply" in condition_path_to_file:
            condition_new_obj = bpy.ops.import_mesh.ply(filepath=condition_path_to_file)
       
        condition_obj_object = bpy.data.objects[str(c_name.replace(".ply", "").replace(".obj", ""))]
       
        condition_mesh = condition_obj_object.data
        for f in condition_mesh.polygons:
            f.use_smooth = True
        
        condition_obj_object.rotation_euler = (math.radians(0), math.radians(0), math.radians(0)) # The default seems 90, 0, 0 while importing .obj into blender 

        if 'conditions_object' in c_name:
            condition_obj_object.active_material = bpy.data.materials.get("yellow")
        else:
            condition_obj_object.active_material = bpy.data.materials.get("green")

    # Analyze files in the folder to distinguish between single-human and multi-human cases
    ori_obj_files = os.listdir(obj_folder)
    ori_obj_files.sort()

    # Dictionaries for storing human and object files
    human_files_dict = collections.defaultdict(list)  # frame_idx -> [human_files]
    object_name_dict = collections.defaultdict(list)  # frame_idx -> [object_files]

    # Regular expressions for matching filenames
    human_pattern = re.compile(r'^(\d{5})(_person\d+)?\.ply$')
    object_pattern = re.compile(r'^(\d{5})_object.*\.ply$')

    for tmp_name in ori_obj_files:
        if ".ply" in tmp_name:
            # Match human files
            human_match = human_pattern.match(tmp_name)
            if human_match:
                frame_idx = int(human_match.group(1))
                human_files_dict[frame_idx].append(tmp_name)
                continue
            
            # Match object files
            object_match = object_pattern.match(tmp_name)
            if object_match:
                frame_idx = int(object_match.group(1))
                object_name_dict[str(frame_idx)].append(tmp_name)

    # Sort files for each frame
    for frame_idx in human_files_dict:
        human_files_dict[frame_idx].sort()

    for frame_idx in object_name_dict:
        object_name_dict[frame_idx].sort()

    # Get all frame indices
    all_frame_indices = sorted(set(human_files_dict.keys()))
    print(f"Found {len(all_frame_indices)} frames with human meshes")

    # Human color configuration
    human_colors = [
        (10/255.0, 30/255.0, 225/255.0, 1),    # Light Blue
        (255/255.0, 100/255.0, 100/255.0, 1),  # Light Red
        (100/255.0, 255/255.0, 100/255.0, 1),  # Light Green
        (255/255.0, 255/255.0, 100/255.0, 1),  # Light Yellow
        (255/255.0, 100/255.0, 255/255.0, 1),  # Light Magenta
        (100/255.0, 255/255.0, 255/255.0, 1),  # Light Cyan
    ]

    # Object color configuration
    object_colors = [
        (153/255.0, 51/255.0, 255/255.0, 1),   # Light Purple
        (255/255.0, 51/255.0, 51/255.0, 1),    # Red
        (51/255.0, 255/255.0, 51/255.0, 1),    # Green
        (255/255.0, 255/255.0, 51/255.0, 1),   # Yellow
        (51/255.0, 255/255.0, 255/255.0, 1),   # Cyan
        (255/255.0, 51/255.0, 255/255.0, 1),   # Magenta
    ]

    for frame_idx in all_frame_indices:
        # if frame_idx % 30 != 0 and frame_idx != all_frame_indices[-1]:
        #     continue
        print(f"Processing frame {frame_idx}")
        
        # Process condition files (if they exist)
        for c_name in condition_files:
            if '_'+str(frame_idx)+'.ply' not in c_name:
                continue
            # if 'condition' not in c_name or int(c_name.split('.')[0].split('_')[-1]) > int(frame_idx):
            #     continue
            condition_path_to_file = os.path.join(condition_folder, c_name)

            # Load human mesh and set material 
            if ".obj" in condition_path_to_file:
                condition_new_obj = bpy.ops.import_scene.obj(filepath=condition_path_to_file, split_mode ="OFF")
            elif ".ply" in condition_path_to_file:
                condition_new_obj = bpy.ops.import_mesh.ply(filepath=condition_path_to_file)
        
            condition_obj_object = bpy.data.objects[str(c_name.replace(".ply", "").replace(".obj", ""))]
        
            condition_mesh = condition_obj_object.data
            for f in condition_mesh.polygons:
                f.use_smooth = True
            
            condition_obj_object.rotation_euler = (math.radians(0), math.radians(0), math.radians(0))  # Default seems to be 90, 0, 0 when importing .obj into Blender
            condition_obj_object.active_material = bpy.data.materials.get("red")

        # Get human file list for current frame
        human_files = human_files_dict.get(frame_idx, [])
        if not human_files:
            print(f"No human files found for frame {frame_idx}")
            continue
            
        print(f"Found {len(human_files)} human files for frame {frame_idx}: {human_files}")
        
        # Store loaded human objects for later deletion
        loaded_human_objects = []
        
        # Load all human meshes
        for human_idx, human_file in enumerate(human_files):
            path_to_file = os.path.join(obj_folder, human_file)
            
            if not os.path.exists(path_to_file):
                print(f"Warning: Human file not found: {path_to_file}")
                continue
            
            # Load human mesh and set material 
            if ".obj" in path_to_file:
                human_new_obj = bpy.ops.import_scene.obj(filepath=path_to_file, split_mode ="OFF")
            elif ".ply" in path_to_file:
                human_new_obj = bpy.ops.import_mesh.ply(filepath=path_to_file)
            
            human_obj_name = human_file.replace(".ply", "").replace(".obj", "")
            human_obj_object = bpy.data.objects[human_obj_name]
            loaded_human_objects.append(human_obj_object)
            
            # Set mesh properties
            human_mesh = human_obj_object.data
            for f in human_mesh.polygons:
                f.use_smooth = True
            
            human_obj_object.rotation_euler = (math.radians(0), math.radians(0), math.radians(0))
            
            # Create material
            human_mat = bpy.data.materials.new(name=f"HumanMaterial_{frame_idx}_{human_idx}")
            human_obj_object.data.materials.append(human_mat)
            human_mat.use_nodes = True
            principled_bsdf = human_mat.node_tree.nodes['Principled BSDF']
            if principled_bsdf is not None:
                # Use different colors for different humans
                color_idx = human_idx % len(human_colors)
                principled_bsdf.inputs[0].default_value = human_colors[color_idx]
                print(f"Set color for human {human_idx} ({human_obj_name}): {human_colors[color_idx]}")
            
            human_obj_object.active_material = human_mat

        # Process object files
        frame_idx_str = str(frame_idx)
        object_files = object_name_dict.get(frame_idx_str, [])
        loaded_object_objects = []
        
        if object_files:
            print(f"Found {len(object_files)} object files for frame {frame_idx}: {object_files}")
            
            for obj_idx, object_file in enumerate(object_files):
                object_path_to_file = os.path.join(obj_folder, object_file)
                
                if not os.path.exists(object_path_to_file):
                    print(f"Warning: Object file not found: {object_path_to_file}")
                    continue
                    
                # Load object mesh and set material 
                if ".obj" in object_path_to_file:
                    new_obj = bpy.ops.import_scene.obj(filepath=object_path_to_file, split_mode ="OFF")
                elif ".ply" in object_path_to_file:
                    new_obj = bpy.ops.import_mesh.ply(filepath=object_path_to_file)
                
                # Get object name
                object_name = object_file.replace(".ply", "").replace(".obj", "")
                obj_object = bpy.data.objects[object_name]
                loaded_object_objects.append(obj_object)
                
                # Set mesh properties
                mesh = obj_object.data
                for f in mesh.polygons:
                    f.use_smooth = True
                
                obj_object.rotation_euler = (math.radians(0), math.radians(0), math.radians(0))
                
                # Create material
                mat = bpy.data.materials.new(name=f"ObjectMaterial_{frame_idx}_{obj_idx}")
                obj_object.data.materials.append(mat)
                mat.use_nodes = True
                principled_bsdf = mat.node_tree.nodes['Principled BSDF']
                if principled_bsdf is not None:
                    # Use different colors for different objects
                    color_idx = obj_idx % len(object_colors)
                    principled_bsdf.inputs[0].default_value = object_colors[color_idx]
                    print(f"Set color for object {obj_idx} ({object_name}): {object_colors[color_idx]}")

                obj_object.active_material = mat

        # Render
        bpy.data.scenes['Scene'].render.filepath = os.path.join(output_dir, ("%05d"%frame_idx)+".jpg")
        bpy.ops.render.render(write_still=True)

        # Clean up unused materials
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

        # Delete all loaded human objects
        for human_obj_object in loaded_human_objects:
            bpy.data.objects.remove(human_obj_object, do_unlink=True)
        
        # Delete all loaded object objects
        for obj_object in loaded_object_objects:
            bpy.data.objects.remove(obj_object, do_unlink=True)

    bpy.ops.wm.quit_blender()