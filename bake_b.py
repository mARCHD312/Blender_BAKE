import argparse
import os
import sys
import time
from math import *
import bpy

from mathutils import *
import subprocess

import collections
'''
New version with rename of models uv layers to map1  06.09.2022.
'''

dir = os.path.join(os.getcwd())
if dir not in sys.path:
    sys.path.append(dir)
from utils import *

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


def create_photo(cwd, obj):
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    baked_photo_name = f"{obj.name}_baked" 
    baked_photo_path = f"{cwd}\\outputs\\{baked_photo_name}.jpeg"
    image_texture_node = obj.active_material.node_tree.nodes.new('ShaderNodeTexImage')
    image = bpy.data.images.new(name=baked_photo_name , width=64, height=64)
    image.filepath_raw = baked_photo_path 
    image.file_format = 'JPEG'
    image.save()

    return image_texture_node, baked_photo_name, baked_photo_path


def delete_unused_nodes():
    for x in range(3):
        for obj in bpy.data.objects:
            if obj.type == "MESH":
                for mat in obj.data.materials:
                    for node in mat.node_tree.nodes:                
                        if node.name == "Material Output": # check this when have time
                            pass
                        else:
                            counter = 0
                            for outp in node.outputs:                        
                                if len(outp.links) >0:
                                    counter = counter + 1                    
                            if counter == 0:
                                mat.node_tree.nodes.remove(node)


def unwrap(mesh, uv_name):
    bpy.ops.mesh.uv_texture_add()
    mesh.data.uv_layers[uv_name].name = "Lmap1"
    change_to_mode("EDIT") 
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(island_margin=0.01)
    change_to_mode("OBJECT")
    deselect_all_in_edit(mesh)


def bake_connect_output(mesh, image_texture_node,baked_photo_name, baked_photo_path):
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
    all_mtls = []
    for obj in bpy.data.objects:
        obj.select_set(True)
        set_object_as_active(obj)
        bpy.ops.object.make_single_user(object=True, obdata=True)
        if obj.type == "MESH":
            all_mtls.append(obj.material_slots[0].name)
    
    mtls_with_more_instances = [item for item, count in collections.Counter(all_mtls).items() if count > 1]
    # mtl_with_one_instance = [mtl for mtl in all_mtls if mtl not in mtls_with_more_instances]

    return mtls_with_more_instances

def GetObjectAndUVMap( objName, uvMapName ):
    try:
        obj = bpy.data.objects[objName]

        if obj.type == 'MESH':
            uvMap = obj.data.uv_layers[uvMapName]
            return obj, uvMap
    except:
        pass

    return None, None


def Scale2D( v, s, p ):
    return ( p[0] + s[0]*(v[0] - p[0]), p[1] + s[1]*(v[1] - p[1]) )     


def ScaleUV( uvMap, scale, pivot ):
    for uvIndex in range( len(uvMap.data) ):
        uvMap.data[uvIndex].uv = Scale2D( uvMap.data[uvIndex].uv, scale, pivot )        

def delete_unused_uv_maps():

    for x in range(2):
        for obj in bpy.data.objects:
            if  obj.type == "MESH":
                try:
                    used_layer = obj.material_slots[0].material.node_tree.nodes['UV Map'].uv_map
                    for uv_layer in obj.data.uv_layers:
                        if uv_layer.name != used_layer:
                            a = obj.data.uv_layers.get(uv_layer.name)
                            obj.data.uv_layers.remove(a)
                except Exception as e:
                    print("Exception with deleteing usused uw maps: ")
                    print(e)            



def prepare_bake_save():
    cwd = os.getcwd()
    deselect_all()
    mtls_with_more_instances =  check_mtls_and_meshes()
       

    # specific_objects = [bpy.data.objects['Box001']]
    #     bpy.data.objects['booth2'],
    #     bpy.data.objects['door3'],
    #     bpy.data.objects['floor_pos_01'],
    #     bpy.data.objects['floor_pos_02'],
    #     bpy.data.objects['floor_pos_03'],
    #     bpy.data.objects['ict_board'],
    #     bpy.data.objects['ict_board1'],
    #     bpy.data.objects['Portal'],
    #     bpy.data.objects['Vault_gymnastics_01']]
    # for obj in specific_objects:  # for specific objects 
    for obj in bpy.data.objects:  # for all objects
        if len(obj.material_slots)<2:
            if  obj.type == "MESH":
                if obj.material_slots[0].name in mtls_with_more_instances:                          
                    change_to_mode("OBJECT")
                    deselect_all()
                    obj.select_set(True)
                    set_object_as_active(obj)
                    image_texture_node, baked_photo_name, baked_photo_path =  create_photo(cwd,obj)
                    name_mtl = f"{obj.name}_mtl"
                    if obj.name.startswith("O"):
                        pass
                    else:
                        uv_names_list = [x.name for x in obj.data.uv_layers]
                        if "UVMap" in uv_names_list:
                            unwrap(obj,'UVMap.001')
                        else:
                            unwrap(obj,'UVMap')
                    image_texture_node = obj.active_material.node_tree.nodes.new('ShaderNodeTexImage')
                    nodes = [n for n in obj.active_material.node_tree.nodes if n.type == 'TEX_IMAGE']
                    try:
                        obj.material_slots[0].material.node_tree.nodes[nodes[-1].name].image = bpy.data.images.load(filepath=baked_photo_path)
                    except:
                        obj.material_slots[0].material.node_tree.nodes["Image Texture"].image = bpy.data.images.load(filepath=baked_photo_path)
                    uv_node_original = obj.active_material.node_tree.nodes.new("ShaderNodeUVMap")
                    try:
                        uv_node_original.uv_map = "map1"
                    except:
                        uv_node_original.uv_map = "UVChannel_1"
                        
                    image_texture_node_original = obj.material_slots[0].material.node_tree.nodes['Image Texture']
                    obj.material_slots[0].material.node_tree.links.new(
                        uv_node_original.outputs['UV'], image_texture_node_original.inputs['Vector'])
                    uv_node = obj.active_material.node_tree.nodes.new("ShaderNodeUVMap")
                    uv_node.uv_map = "Lmap1"
                    obj.material_slots[0].material.node_tree.links.new(
                        uv_node.outputs['UV'], image_texture_node.inputs['Vector'])
                    image_texture_node.select = True
                    obj.active_material.node_tree.nodes.active = image_texture_node                
                    try:
                        bpy.ops.object.bake(type='COMBINED')         
                    except:
                        pass                     


                     
                    bpy.data.images[f"{baked_photo_name}.jpeg"].save_render(baked_photo_path)
                    mat = bpy.data.materials.new(name=name_mtl)
                    obj.data.materials.append(mat)
                    new_mtl_index = len(obj.material_slots) - 1
                    obj.active_material_index = new_mtl_index
                    obj.active_material.use_nodes = True
                    mat.node_tree.nodes.remove(mat.node_tree.nodes["Principled BSDF"])
                    image_texture_node_finish = obj.active_material.node_tree.nodes.new('ShaderNodeTexImage')
                    obj.material_slots[new_mtl_index].material.node_tree.nodes['Image Texture'].image = bpy.data.images.load(filepath=baked_photo_path)
                    uv_node = obj.active_material.node_tree.nodes.new("ShaderNodeUVMap")
                    uv_node.uv_map = "Lmap1"
                    obj.material_slots[new_mtl_index].material.node_tree.links.new(
                        uv_node.outputs['UV'], image_texture_node_finish.inputs['Vector'])
                    output = obj.material_slots[new_mtl_index].material.node_tree.nodes['Material Output']
                    obj.material_slots[new_mtl_index].material.node_tree.links.new(
                        image_texture_node_finish.outputs['Color'], output.inputs['Surface'])
                    
                    for x in range(new_mtl_index):
                        obj.data.materials.pop(index = 0)

                else:
                    try:              
                        deselect_all()
                        obj.select_set(True)
                        set_object_as_active(obj)
                        image_texture_node, baked_photo_name, baked_photo_path =  create_photo(cwd,obj)
                        nodes = [n for n in obj.active_material.node_tree.nodes if n.type == 'TEX_IMAGE']
                        obj.material_slots[0].material.node_tree.nodes[nodes[-1].name].image = bpy.data.images.load(                    
                        filepath=baked_photo_path)
                        if obj.name.startswith("O"):
                            pass
                        else:
                            unwrap(obj,"UVMap")      
                        bake_connect_output(obj,image_texture_node,baked_photo_name,baked_photo_path)
                        
                    except Exception as e:
                        print(e)
                        

        else:
            change_to_mode("OBJECT")
            deselect_all()
            obj.select_set(True)
            set_object_as_active(obj)
            image_texture_node, baked_photo_name, baked_photo_path =  create_photo(cwd,obj)
            name_mtl = f"{obj.name}_mtl"
            uv_names_list = [x.name for x in obj.data.uv_layers]

            if "UVMap" in uv_names_list:
                unwrap(obj,'UVMap.001')
            else:
                unwrap(obj,'UVMap')

            for count, mtl in enumerate(obj.material_slots):
                obj.active_material_index = count
                image_texture_node = obj.active_material.node_tree.nodes.new('ShaderNodeTexImage')
                nodes = [n for n in obj.active_material.node_tree.nodes if n.type == 'TEX_IMAGE']
                obj.active_material.node_tree.nodes[nodes[-1].name].image = bpy.data.images.load(filepath=baked_photo_path)
                uv_node_original = obj.active_material.node_tree.nodes.new("ShaderNodeUVMap")
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
            image_texture_node_finish = obj.active_material.node_tree.nodes.new('ShaderNodeTexImage')
            obj.material_slots[new_mtl_index].material.node_tree.nodes['Image Texture'].image = bpy.data.images.load(filepath=baked_photo_path)
                    
            uv_node = obj.active_material.node_tree.nodes.new("ShaderNodeUVMap")
            uv_node.uv_map = "Lmap1"
            obj.material_slots[new_mtl_index].material.node_tree.links.new(
                uv_node.outputs['UV'], image_texture_node_finish.inputs['Vector'])
            
            output = obj.material_slots[new_mtl_index].material.node_tree.nodes['Material Output']
            obj.material_slots[new_mtl_index].material.node_tree.links.new(
            image_texture_node_finish.outputs['Color'], output.inputs['Surface'])
                    
            for x in range(new_mtl_index):
                obj.data.materials.pop(index = 0)



def createing_atlas():

    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    x_axis_pivots = [0,0.333,0.666,1]
    y_axis_pivots = [0,0.333,0.666,1]

    coo_uv_meshes = []
    for x in x_axis_pivots:
        for y in y_axis_pivots:
            a = [x,y]
            coo_uv_meshes.append(a)

    for count ,obj in enumerate(meshes[:16]):
        deselect_all()
        set_object_as_active(obj)
        obj.select_set(True)
        scale = Vector((0.25, 0.25))
        pivot = Vector((coo_uv_meshes[count][0], coo_uv_meshes[count][1]))

        #Get the object from names
        obj, uvMap = GetObjectAndUVMap( obj.name, "Lmap1" )

        #If the object is found, scale its UV map
        if obj is not None:
            ScaleUV( uvMap, scale, pivot )
    
    for obj in meshes[:16]:
        obj.select_set(True)

    bpy.ops.object.join()
    cwd = os.getcwd()
    subprocess.run(['python',f"{cwd}\\merge_images.py"], check=True)
    for x in range(len(obj.material_slots) - 1):
        obj.data.materials.pop(index = 0)
    
    nodes = [n for n in obj.active_material.node_tree.nodes if n.type == 'TEX_IMAGE']
    nodes[0].image = bpy.data.images.load(filepath= f"{cwd}\\merged_image.jpg")

def rename_uv_layers():
    for obj in bpy.data.objects:
        try:
            obj.data.uv_layers[0].name = 'map1'
        # except e as Exception:
            # print(e)
        except:
            pass

def main():
    start_time = time.time()
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.cycles.adaptive_threshold = 0.9
    rename_uv_layers()
    # raise
    print("BAke start")
    prepare_bake_save()

    delete_unused_nodes()
    delete_unused_uv_maps()
    # raise
    # createing_atlas()    
    end_time = time.time()
    save_to_blend_file(SAVE_FILE)
    bpy.ops.export_scene.gltf(filepath=SAVE_FILE[:-5] + 'gltf')
    print("finished in: ")
    print(end_time-start_time)

main()


