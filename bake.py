import argparse
import os
import sys
import time
from math import *
import bpy
from mathutils import *
import collections


dir = os.path.join(os.getcwd())
if dir not in sys.path:
    sys.path.append(dir)

PARSER = argparse.ArgumentParser(description="Baker")
PARSER.add_argument(
    '-s',
    '--savefile',
    help='Location of the .BLEND file',
    required=True
)
PARSER.add_argument(
    '-p',
    '--project_dir',
    help='Location of project',
    required=True,
)


ARGS = PARSER.parse_args(sys.argv[sys.argv.index("--") + 1:])
SAVE_FILE = ARGS.savefile
PROJECT_FOLDER = ARGS.project_dir
sys.path.append(os.path.join(PROJECT_FOLDER))
from utils import *

def create_photo(cwd, obj):

    """
    Creates a blank baked photo and a corresponding image texture node.

    :param cwd: current working directory
    :type cwd: str
    :param obj: object to bake
    :type obj: bpy.types.Object
    :return: image texture node, baked photo name, baked photo path
    :rtype: tuple
    """
    
    bpy.context.scene.render.image_settings.file_format = 'JPEG'  # set the format of the image
    baked_photo_name = f"{obj.name}_baked" # naming the photo
    baked_photo_path = os.path.join(cwd, "outputs",f"{baked_photo_name}.jpeg") # joining the path
    image_texture_node = obj.active_material.node_tree.nodes.new(
        'ShaderNodeTexImage')  # creating a new image texture node

    image = bpy.data.images.new(name=baked_photo_name, width=2048, height=2048)
    image.filepath_raw = baked_photo_path
    image.file_format = 'JPEG'
    image.save()

    return image_texture_node, baked_photo_name, baked_photo_path


def delete_unused_nodes():
    """
    Deletes unused nodes from the node tree of all materials of all mesh objects.

    :return: None
    """
    for obj in bpy.data.objects:
        if obj.type == "MESH": # check if the object is a mesh
            for mat in obj.data.materials: #iterating through all materials
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                used_nodes = set()
                for link in links: # creating a set of used nodes
                    used_nodes.add(link.from_node)
                    used_nodes.add(link.to_node)
                for node in nodes: # iterating through all nodes
                    if node not in used_nodes: # checking if the node is not used
                        nodes.remove(node) # remove the node from the node tree


def unwrap(mesh, uv_name):
    """
    Unwraps the UVs of a mesh.
    
    :param mesh: mesh to unwrap
    :type mesh: bpy.types.Mesh
    :param uv_name: name of the UV map
    :type uv_name: str
    :return: None
    """
    bpy.ops.mesh.uv_texture_add() # add a new UV map
    mesh.data.uv_layers[uv_name].name = "Lmap1" # rename the UV map
    change_to_mode("EDIT") # change to edit mode
    bpy.ops.mesh.select_all(action='SELECT') # select all faces
    bpy.ops.uv.smart_project(island_margin=0.01) # smart project the UVs
    change_to_mode("OBJECT") # change back to object mode
    deselect_all_in_edit(mesh) # deselect all faces


def bake_connect_output(mesh, image_texture_node, baked_photo_name, baked_photo_path):
    """
    Bakes the object and connects the output to the image texture node.

    :param mesh: mesh to bake
    :type mesh: bpy.types.Mesh
    :param image_texture_node: image texture node to connect the output to
    :type image_texture_node: bpy.types.ShaderNodeTexImage
    :param baked_photo_name: name of the baked photo
    :type baked_photo_name: str
    :param baked_photo_path: path of the baked photo
    :type baked_photo_path: str
    :return: None
    """
    bpy.context.object.active_material.node_tree.nodes.active = image_texture_node
    bpy.ops.object.bake(type='COMBINED')
    output = mesh.material_slots[0].material.node_tree.nodes['Material Output']
    mesh.material_slots[0].material.node_tree.links.new(
        image_texture_node.outputs['Color'], output.inputs['Surface'])

    obj = bpy.context.active_object
    uv_node = obj.active_material.node_tree.nodes.new("ShaderNodeUVMap")
    uv_node.uv_map = "Lmap1"
    obj.material_slots[len(obj.material_slots) - 1].material.node_tree.links.new(
        uv_node.outputs['UV'], image_texture_node.inputs['Vector'])

    bpy.data.images[f"{baked_photo_name}.jpeg"].save_render(baked_photo_path)


def check_mtls_and_meshes():
    """
    Check for materials that are used on more than one object.

    :return: A list of materials that are used on more than one object
    :rtype: list
    """
    all_mtls = []
    for obj in bpy.data.objects:
        obj.select_set(True)
        set_object_as_active(obj)
        bpy.ops.object.make_single_user(object=True, obdata=True)
        if obj.type == "MESH":
            all_mtls.append(obj.material_slots[0].name)

    mtls_with_more_instances = [
        item for item, count in collections.Counter(all_mtls).items() if count > 1]

    return mtls_with_more_instances


def delete_uv_maps_of_objects_that_starts_with_O():
    
    """
    Deletes the UV maps of objects that starts with 'O'.

    :return: None
    """
    
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name.startswith("O"):
            try:
                # get the used uv layer
                used_layer = obj.material_slots[0].material.node_tree.nodes['UV Map'].uv_map
                # get the uv layer by name and remove it
                layer = obj.data.uv_layers.get(used_layer.name)
                obj.data.uv_layers.remove(layer)
            except Exception as ex:
                print("Exception with deleteing usused uw maps: ")
                print(ex)

            for mat in obj.data.materials:
                for node in mat.node_tree.nodes:
                    if node.name == "UV Map":
                        mat.node_tree.nodes.remove(node)

def rename_uv_layers():

    """
    Rename all the UV maps of objects'.

    :return: None
    """

    for obj in bpy.data.objects:
        if obj.type == "MESH":
            try:
                obj.data.uv_layers[0].name = 'map1'
            except Exception as ex:
                print("Exception while renameing model uv layer:")
                print(ex)



def prepare_bake_save():
    cwd = os.getcwd()
    deselect_all()
    mtls_with_more_instances = check_mtls_and_meshes()

    # for every objecty in coollection
    # for obj in bpy.data.collections['for_bake'].objects:
    for obj in bpy.data.objects:
        # # if object have less than two material slots and is mesh tye
        if len(obj.material_slots) < 2:
            if obj.type == "MESH":
                print("ONE MTL, OBJECT: ")
                print(obj.name)
    #             # if material name in materiales with more instances
                if obj.material_slots[0].name in mtls_with_more_instances:
                    change_to_mode("OBJECT")
                    deselect_all()
                    obj.select_set(True)
                    set_object_as_active(obj)
                    # Create a new image texture node, baked_photo_name and baked_photo_path
                    image_texture_node, baked_photo_name, baked_photo_path = create_photo(
                        cwd, obj)
                    name_mtl = f"{obj.name}_mtl"
                     # if the object's name doesn't start with 'O'
                    if not obj.name.startswith("O"):
                        # get a list of the names of the object's uv maps
                        uv_names_list = [x.name for x in obj.data.uv_layers]
                        if "UVMap" in uv_names_list:
                            # unwrap the object's mesh using the UV map named "UVMap.001"
                            unwrap(obj, 'UVMap.001')
                        else:
                            # unwrap the object's mesh using the UV map named "UVMap"
                            unwrap(obj, 'UVMap')

                    image_texture_node = obj.active_material.node_tree.nodes.new(
                        'ShaderNodeTexImage')
                    nodes = [
                        n for n in obj.active_material.node_tree.nodes if n.type == 'TEX_IMAGE']
                    try:
                        obj.material_slots[0].material.node_tree.nodes[nodes[-1].name].image = bpy.data.images.load(
                            filepath=baked_photo_path)
                    except Exception as ex:
                        print(ex)
                        obj.material_slots[0].material.node_tree.nodes["Image Texture"].image = bpy.data.images.load(
                            filepath=baked_photo_path)
                    uv_node_original = obj.active_material.node_tree.nodes.new(
                        "ShaderNodeUVMap")
                    try:
                        uv_node_original.uv_map = "map1"
                    except Exception as ex:
                        print(ex)
                        uv_node_original.uv_map = "UVChannel_1"
                    image_texture_node_original = obj.material_slots[
                        0].material.node_tree.nodes['Image Texture']
                    obj.material_slots[0].material.node_tree.links.new(
                        uv_node_original.outputs['UV'], image_texture_node_original.inputs['Vector'])
                    uv_node = obj.active_material.node_tree.nodes.new(
                        "ShaderNodeUVMap")
                    uv_node.uv_map = "Lmap1"
                    obj.material_slots[0].material.node_tree.links.new(
                        uv_node.outputs['UV'], image_texture_node.inputs['Vector'])
                    image_texture_node.select = True
                    obj.active_material.node_tree.nodes.active = image_texture_node
                    try:
                        bpy.ops.object.bake(type='COMBINED')
                    except Exception as ex:
                        print(ex)

                    bpy.data.images[f"{baked_photo_name}.jpeg"].save_render(
                        baked_photo_path)
                    mat = bpy.data.materials.new(name=name_mtl)
                    obj.data.materials.append(mat)
                    new_mtl_index = len(obj.material_slots) - 1
                    obj.active_material_index = new_mtl_index
                    obj.active_material.use_nodes = True
                    mat.node_tree.nodes.remove(
                        mat.node_tree.nodes["Principled BSDF"])
                    image_texture_node_finish = obj.active_material.node_tree.nodes.new(
                        'ShaderNodeTexImage')
                    obj.material_slots[new_mtl_index].material.node_tree.nodes['Image Texture'].image = bpy.data.images.load(
                        filepath=baked_photo_path)
                    uv_node = obj.active_material.node_tree.nodes.new(
                        "ShaderNodeUVMap")
                    uv_node.uv_map = "Lmap1"
                    obj.material_slots[new_mtl_index].material.node_tree.links.new(
                        uv_node.outputs['UV'], image_texture_node_finish.inputs['Vector'])
                    output = obj.material_slots[new_mtl_index].material.node_tree.nodes['Material Output']
                    obj.material_slots[new_mtl_index].material.node_tree.links.new(
                        image_texture_node_finish.outputs['Color'], output.inputs['Surface'])

                    for x in range(new_mtl_index):
                        obj.data.materials.pop(index=0)

                else:
                    try:
                        deselect_all()
                        obj.select_set(True)
                        set_object_as_active(obj)
                        image_texture_node, baked_photo_name, baked_photo_path = create_photo(
                            cwd, obj)
                        nodes = [
                            n for n in obj.active_material.node_tree.nodes if n.type == 'TEX_IMAGE']
                        obj.material_slots[0].material.node_tree.nodes[nodes[-1].name].image = bpy.data.images.load(
                            filepath=baked_photo_path)
                        if not obj.name.startswith("O"):
                            uv_names_list = [
                                x.name for x in obj.data.uv_layers]
                            if "UVMap" in uv_names_list:
                                unwrap(obj, 'UVMap.001')
                            else:
                                unwrap(obj, 'UVMap')

                        bake_connect_output(
                            obj, image_texture_node, baked_photo_name, baked_photo_path)

                    except Exception as ex:
                        print(ex)

        else:
            print("MORE MTLS, OBJECT:")
            print(obj.name)
            change_to_mode("OBJECT")
            deselect_all()
            obj.select_set(True)
            set_object_as_active(obj)
            image_texture_node, baked_photo_name, baked_photo_path = create_photo(
                cwd, obj)
            name_mtl = f"{obj.name}_mtl"
            uv_names_list = [x.name for x in obj.data.uv_layers]

            if not obj.name.startswith("O"):
                if "UVMap" in uv_names_list:
                    unwrap(obj, 'UVMap.001')
                else:
                    unwrap(obj, 'UVMap')

            for count, mtl in enumerate(obj.material_slots):
                obj.active_material_index = count
                image_texture_node = obj.active_material.node_tree.nodes.new(
                    'ShaderNodeTexImage')
                nodes = [
                    n for n in obj.active_material.node_tree.nodes if n.type == 'TEX_IMAGE']
                obj.active_material.node_tree.nodes[nodes[-1].name].image = bpy.data.images.load(
                    filepath=baked_photo_path)
                uv_node_original = obj.active_material.node_tree.nodes.new(
                    "ShaderNodeUVMap")
                uv_node_original.uv_map = "map1"
                obj.material_slots[count].material.node_tree.links.new(
                    uv_node_original.outputs['UV'], image_texture_node.inputs['Vector'])
                obj.active_material.node_tree.nodes.active = image_texture_node

                image_texture_node.image = bpy.data.images[f"{obj.name}_baked"]
            bpy.ops.object.bake(type='COMBINED')
            bpy.data.images[f"{obj.name}_baked"].save_render(baked_photo_path)
            mat = bpy.data.materials.new(name=name_mtl)
            obj.data.materials.append(mat)

            new_mtl_index = len(obj.material_slots) - 1
            obj.active_material_index = new_mtl_index
            obj.active_material.use_nodes = True
            mat.node_tree.nodes.remove(mat.node_tree.nodes["Principled BSDF"])
            image_texture_node_finish = obj.active_material.node_tree.nodes.new(
                'ShaderNodeTexImage')
            obj.material_slots[new_mtl_index].material.node_tree.nodes['Image Texture'].image = bpy.data.images.load(
                filepath=baked_photo_path)

            uv_node = obj.active_material.node_tree.nodes.new(
                "ShaderNodeUVMap")
            uv_node.uv_map = "Lmap1"
            obj.material_slots[new_mtl_index].material.node_tree.links.new(
                uv_node.outputs['UV'], image_texture_node_finish.inputs['Vector'])

            output = obj.material_slots[new_mtl_index].material.node_tree.nodes['Material Output']
            obj.material_slots[new_mtl_index].material.node_tree.links.new(
                image_texture_node_finish.outputs['Color'], output.inputs['Surface'])

            for x in range(new_mtl_index):
                obj.data.materials.pop(index=0)

def main():
    start_time = time.time()
    print("Bake started")
    bpy.context.scene.cycles.samples = 10
    bpy.context.scene.cycles.adaptive_threshold = 0.05
    rename_uv_layers()
    prepare_bake_save()
    delete_unused_nodes()
    delete_uv_maps_of_objects_that_starts_with_O()
    save_to_blend_file(SAVE_FILE)
    bpy.ops.export_scene.gltf(filepath=SAVE_FILE[:-5] + 'gltf')
    end_time = time.time()
    print("finished in: ")
    print(end_time-start_time)


main()
