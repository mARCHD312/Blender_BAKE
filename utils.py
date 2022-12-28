import bpy
import bmesh
import time
import pathlib
from mathutils import *
from mathutils.bvhtree import BVHTree
from math import *
import sys
import os
import numpy as np


def delete_all_objects():
    change_to_mode('OBJECT')
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)


def change_to_mode(mode):
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode=mode)


def deselect_all():
    bpy.ops.object.select_all(action='DESELECT')


def set_object_as_active(obj):
    bpy.context.view_layer.objects.active = obj


def deselect_all_in_edit(mesh):
    bpy.ops.object.mode_set(mode="OBJECT")

    # vertices can be selected
    # to deselect vertices you need to deselect faces(polygons)
    #  and edges at first
    for i in mesh.data.polygons:
        i.select = False
    for i in mesh.data.edges:
        i.select = False
    for i in mesh.data.vertices:
        i.select = False


def delete_object(obj):
    change_to_mode('OBJECT')
    bpy.data.objects.remove(obj)


def read_command_line_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after --
    return argv


def set_frame(frame_num):
    bpy.context.scene.frame_set(frame_num)


def get_mesh():
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            mesh = obj
            break
    return mesh


def get_mesh_eval(mesh):
    return mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())


def load_model(filepath):
    suffix = pathlib.Path(filepath).suffix
    if suffix == ".fbx":
        load_fbx(filepath)
    elif suffix == ".blend":
        load_blend(filepath)
    elif suffix == '.obj' or suffix == '.OBJ':
        load_obj(filepath)
    else:
        print("unsupported file model")


def load_fbx(filepath):
    bpy.ops.import_scene.fbx(filepath=filepath)


def load_blend(filepath):
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects]
    for obj in data_to.objects:
        bpy.context.collection.objects.link(obj)


def load_obj(filepath):
    bpy.ops.import_scene.obj(filepath=filepath)


def save_to_blend_file(filepath):
    change_to_mode('OBJECT')
    bpy.ops.wm.save_as_mainfile(filepath=filepath)


def apply_scale_and_rotation(mesh):
    change_to_mode('OBJECT')
    deselect_all()
    mesh.select_set(True)
    set_object_as_active(mesh)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)


def apply_rotation(mesh):
    change_to_mode('OBJECT')
    deselect_all()
    mesh.select_set(True)
    set_object_as_active(mesh)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)


def get_overlaping_points(bvh1, bvh2):
    overlap_values = bvh1.overlap(bvh2)
    if overlap_values:
        overlap = True
    else:
        overlap = False
    return overlap, overlap_values


def refresh_frame():
    frame = bpy.context.scene.frame_current
    bpy.context.scene.frame_set(1)
    bpy.context.scene.frame_set(frame)


def bvh_tree(obj):
    mat = obj.matrix_world
    vert = [mat @ v.co for v in obj.data.vertices]
    poly = [p.vertices for p in obj.data.polygons]
    bvh = BVHTree.FromPolygons(vert, poly)
    return bvh


def add_sphere(location, radius=0.01, name=None):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    sphere = bpy.context.active_object
    if name is not None:
        sphere.name = name
    return bpy.context.active_object


def add_plane(start_pos, width, rotation=(0, 0, 0)):
    cursor = bpy.context.scene.cursor
    cursor.location = start_pos
    # add plane on cursor location
    bpy.ops.mesh.primitive_plane_add(size=width,
                                     calc_uvs=True,
                                     enter_editmode=False,
                                     align='WORLD',
                                     location=(start_pos),
                                     rotation=rotation)
    plane = bpy.context.active_object
    return plane


def add_new_mesh(name, verts, faces, edges=[], col_name="Collection"):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(mesh.name, mesh)
    col = bpy.data.collections.get(col_name)
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    mesh.from_pydata(verts, edges, faces)
    return obj


def decimate_object(obj, count=25000):
    change_to_mode('OBJECT')
    deselect_all()
    obj.select_set(True)
    set_object_as_active(obj)
    bpy.ops.object.modifier_add(type='DECIMATE')
    bpy.context.object.modifiers["Decimate"].ratio = count / \
        len(obj.data.vertices)
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Decimate")


def subdivide_mesh(mesh, n_cuts=8):
    deselect_all()
    mesh.select_set(True)
    change_to_mode('EDIT')
    bpy.ops.mesh.subdivide(number_cuts=n_cuts)
    change_to_mode('OBJECT')


def change_to_basis(vector, new_origin, new_basis):
    vector_wrt_new_origin = vector - new_origin
    # mat = Matrix((leftArm.x_axis, leftArm.y_axis,
    # leftArm.z_axis)).transposed()
    local_coord = new_basis.inverted() @ vector_wrt_new_origin
    return local_coord


def list_all_zeros(l):
    all_zeros = True
    for el in l:
        if el != 0:
            all_zeros = False
            break
    return all_zeros


def delete_polygons_from_mesh(mesh, poly_indexes):
    change_to_mode('OBJECT')
    deselect_all()
    set_object_as_active(mesh)

    faces = [mesh.data.polygons[i] for i in poly_indexes]
    for f in faces:
        f.select = True

    change_to_mode('EDIT')
    bpy.ops.mesh.delete(type='FACE')
    change_to_mode('OBJECT')


def project_with_direction(vectorFrom, vectorTo):
    vector_proj = vectorFrom.project(vectorTo)
    angle = vector_proj.angle(vectorTo)
    dir = 1 if angle <= pi/2 else -1
    return dir * (vector_proj).length


def rotate_bone_to_tail_position(armature, bone, tail_position):
    mat_w = armature.matrix_world
    vec1 = mat_w @ bone.vector
    vec2 = tail_position - mat_w @ bone.head
    rot_diff = vec1.rotation_difference(vec2)
    R = rot_diff.to_matrix().to_4x4()
    bone.matrix = R @ bone.matrix
    return R


def rotate_bone_and_insert_keyframe(armature, bone, tail_position):
    rotate_bone_to_tail_position(armature, bone, tail_position)
    bone.keyframe_delete(data_path="rotation_quaternion")
    bone.keyframe_insert(data_path="rotation_quaternion")
    refresh_frame()


def apply_rotation_and_insert_keyframe(bone, rotation):
    bone.rotation_quaternion = rotation.copy()
    bone.keyframe_delete(data_path="rotation_quaternion")
    bone.keyframe_insert(data_path="rotation_quaternion")
    refresh_frame()


def from_vector_to_array(vector):
    array_vec = [vector.x, vector.y, vector.z]
    return array_vec


def from_array_to_vector(array_vec):
    vec = Vector((array_vec[0], array_vec[1], array_vec[2]))
    return vec


def from_quat_to_arr(quat):
    return [quat[0], quat[1], quat[2], quat[3]]


def from_arr_to_quat(arr):
    return Quaternion((arr[0], arr[1], arr[2], arr[3]))

# Bones and its parents


def BoneAndParents(bone):
    yield bone
    while bone.parent:
        bone = bone.parent
        yield bone

# Move bone with inverse kinematic


def ik_bone_to_coordinate(bone, coordinates, chain_len, use_tail):
    arm = bpy.data.objects['Armature']
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.context.scene.cursor.location = coordinates

    bpy.ops.object.mode_set(mode='OBJECT')

    # Add a temporary empty
    bpy.ops.object.add()
    empty = bpy.context.view_layer.objects.active

    bpy.context.view_layer.objects.active = arm

    bpy.ops.object.mode_set(mode='POSE')

    bone = arm.pose.bones[bone]

    # Add IK constraint and update
    constraint = bone.constraints.new('IK')
    constraint.target = empty
    bone.constraints['IK'].chain_count = chain_len
    bone.constraints['IK'].use_tail = use_tail

    bone.constraints.update()

    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()

    # bpy.context.scene.update() #Important: update the scene

    # Get the bones and transformation due to the constraint
    bones = [b for b in BoneAndParents(bone)]
    transformations = [b.matrix.copy() for b in bones]

    # Get rid of the temporary constraint and update
    bone.constraints.remove(constraint)
    bone.constraints.update()

    # Assign transformation
    for b, t in zip(bones, transformations):
        b.matrix = t

    # Remove temporary empty
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = empty
    bpy.ops.object.delete()

    # Back to pose
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')

    bpy.ops.object.mode_set(mode='OBJECT')


def is_point_in_mesh(p, obj):
    p_mesh = obj.matrix_world.inverted() @ p

    hit, point, normal, face = obj.closest_point_on_mesh(p)
    if not hit:
        return False
    p2 = point - p_mesh
    v = p2.dot(normal)
    return not(v < 0.0)


def get_orthogonal_unit_vector(vector):
    ort = Vector((0, 0, 0))

    for m in range(3):
        if vector[m] != 0.0:
            break

    for n in range(3):
        if n != m:
            break

    ort[n] = vector[m]
    ort[m] = -vector[n]

    ort.normalize()
    return ort


def find_closest_poly(mesh, location):
    mat_w = mesh.matrix_world
    c_p = min(mesh.data.polygons, key=lambda p: (
        (mat_w @ p.center) - location).length)
    return c_p


def get_edges_key_to_polys(edge_keys, polygons):
    edges_key_to_polys = {}
    for poly in polygons:
        for edge_key in poly.edge_keys:
            if edge_key not in edges_key_to_polys:
                edges_key_to_polys[edge_key] = []
            edges_key_to_polys[edge_key].append(poly)
    return edges_key_to_polys


def get_neighb_of_poly(poly, edges_key_to_polys):
    neighb = []
    for e_key in poly.edge_keys:
        for p in edges_key_to_polys[e_key]:
            if p != poly:
                neighb.append(p)
    return neighb


def travel_through_polys(mesh,
                         edges_key_to_polys,
                         start_poly,
                         poly_indexes_to_skip):
    stack = [start_poly]

    visited = {}
    while len(stack) > 0:
        poly = stack.pop()
        visited[poly.index] = True
        neigbors = get_neighb_of_poly(poly, edges_key_to_polys)
        for n_p in neigbors:
            if (n_p.index not in visited) and (
                n_p.index not in poly_indexes_to_skip):
                stack.append(n_p)

    return visited.keys()


def add_camera(location, rotation):
    bpy.ops.object.camera_add(
        enter_editmode=False, align='VIEW',
        location=location, rotation=rotation)
    camera = bpy.context.active_object
    bpy.context.scene.render.resolution_x = 1080
    bpy.context.scene.render.resolution_x = 1920

    return camera


def raycast_along_plane(origin, plane_normal, n_raycasts):
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    plane_normal = plane_normal.normalized()
    x_vec = get_orthogonal_unit_vector(plane_normal)
    y_vec = x_vec.cross(plane_normal).normalized()

    angles = np.linspace(0, 2 * pi, num=n_raycasts)
    angles_deg = [degrees(angle) for angle in angles]
    # print(angles_deg)
    locations = []
    for angle in angles:
        destination = origin + x_vec * sin(angle) + y_vec * cos(angle)
        direction = (destination - origin).normalized()
        result, location, _, _, _, _ = scene.ray_cast(
            view_layer, origin, direction)
        locations.append(location)

    return locations


def find_orthogonal_plane(origin, precision=20, n_raycasts=16):
    start_dir = Vector((1, 0, 0))
    y_angles = np.linspace(0, pi, num=precision)
    z_angles = np.linspace(0, pi, num=precision)

    directions = []
    for y in y_angles:
        for z in z_angles:
            direction = start_dir.copy()
            rotation = Euler((0, y, z))
            direction.rotate(rotation)
            directions.append(direction.copy())

    distances = [0] * len(directions)
    for (i, direction) in enumerate(directions):

        locations = raycast_along_plane(origin, direction, n_raycasts)
        n_half = int(len(locations) / 2)
        location_pairs = [(locations[j], locations[j + n_half])
                          for j in range(n_half)]
        distances_pairs = [((origin - l[0]).length, (origin - l[1]).length)
                           for l in location_pairs]
        all_lengths = [d_p[0] + d_p[1] for d_p in distances_pairs]
        max_index = max(range(len(all_lengths)), key=lambda j: all_lengths[j])
        new_origin = (location_pairs[max_index]
                      [0] + location_pairs[max_index][1]) / 2
        locations = raycast_along_plane(new_origin, direction, n_raycasts)
        loc_dist = [(new_origin - l).length for l in locations]
        max_dist = max(loc_dist)
        distances[i] = max_dist

    index_min = min(range(len(distances)), key=lambda i: distances[i])
    min_direction = directions[index_min]

    return min_direction


def point_at(obj, target, roll=0):
    if not isinstance(target, Vector):
        target = Vector(target)
    loc = obj.location
    direction = target - loc
    quat = direction.to_track_quat('Z', 'Y')
    quat = quat.to_matrix().to_4x4()
    rollMatrix = Matrix.Rotation(roll, 4, 'Z')

    loc = loc.to_tuple()
    obj.matrix_world = quat @ rollMatrix
    obj.location = loc


def add_arrow(location, vector_direction, size=0.1):
    point_vector = Vector(location) + vector_direction
    bpy.ops.object.empty_add(type='SINGLE_ARROW', location=location)
    arrow = bpy.context.object
    point_at(arrow, point_vector)
    arrow.scale[0] = size
    arrow.scale[1] = size
    arrow.scale[2] = size


def average_vectors(vectors):
    vec_sum = None
    for vec in vectors:
        vec_sum = vec if vec_sum is None else vec_sum + vec

    return vec_sum / len(vectors)


# returns locations of collisions
def ray_cast_until_no_collision(origin, direction):
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    result = True
    locations = []
    n = 0
    while result:
        result, location, _, _, _, _ = scene.ray_cast(
            view_layer, origin, direction)
        if result:
            n += 1
            locations.append(location.copy())
            # to avoid colliding with the same point
            origin = location.copy() + 0.001 * direction

    return locations


def cut_ortohogonal_with_raycast(mesh: bpy.types.Mesh, origin, direction, n_raycasts):
    mat_w = mesh.matrix_world
    mat_w_i = mesh.matrix_world.inverted()

    direction = direction.normalized()
    x_vec = get_orthogonal_unit_vector(direction)
    y_vec = x_vec.cross(direction).normalized()

    poly_indexes = set()
    avg_loc = None
    for i in range(n_raycasts):

        angle = i * (2 * pi / n_raycasts)

        x_magn = sin(angle)
        y_magn = cos(angle)

        ray_cast_pos = origin.copy()
        ray_cast_pos += x_magn * x_vec
        ray_cast_pos += y_magn * y_vec

        cast_origin = mat_w_i @ origin
        cast_direction = mat_w_i @ ((ray_cast_pos - origin).normalized())
        result, location, _, index = mesh.ray_cast(cast_origin, cast_direction)
        if result:
            poly_indexes.add(index)
            loc = mat_w @ location
            avg_loc = loc if avg_loc is None else avg_loc + loc

    avg_loc /= n_raycasts

    return poly_indexes, avg_loc


def add_empty(location, radius=0.01, name=None):
    bpy.ops.object.empty_add(location=location, radius=radius)
    empty = bpy.context.active_object
    if name is not None:
        empty.name = name
    return empty


def get_closest_poly_to_loc(mesh, location):
    list_off_all_distances_1 = []
    for poly in mesh.data.polygons:
        final_co = mesh.matrix_world @ poly.center
        p1 = np.array([final_co[0], final_co[1], final_co[2]])
        p2 = np.array([location[0], location[1], location[2]])
        squared_dist = np.sum((p1-p2)**2, axis=0)
        dist = np.sqrt(squared_dist)
        list_off_all_distances_1.append(dist)

    min_dist_1 = min(list_off_all_distances_1)

    for poly in mesh.data.polygons:
        final_co = mesh.matrix_world @ poly.center
        p1 = np.array([final_co[0], final_co[1], final_co[2]])
        p2 = np.array([location[0], location[1], location[2]])
        squared_dist = np.sum((p1-p2)**2, axis=0)
        dist = np.sqrt(squared_dist)
        if (dist == min_dist_1):
            closest_1 = poly
    index_closest_one = closest_1.index
    return index_closest_one


def ray_cast_orig_dest(origin, destination):
    distance = (destination - origin).length
    direction = (destination - origin).normalized()
    result, location, normal, index, obj, matrix = bpy.context.scene.ray_cast(
        bpy.context.view_layer, origin, direction, distance=distance)
    return result, location, normal, index, obj, matrix


def deselect_all_edit(mesh):
    bpy.ops.object.mode_set(mode="OBJECT")
    # vertices can be selected
    # to deselect vertices you need to deselect 
    # faces(polygons) and edges at first
    for i in mesh.data.polygons:
        i.select = False
    for i in mesh.data.edges:
        i.select = False
    for i in mesh.data.vertices:
        i.select = False
    # bpy.ops.object.mode_set(mode="EDIT")


def get_sphere_directions(precision=10):
    precision = 20

    directions = []
    start_dir = Vector((1, 0, 0))
    z_angles = np.linspace(0, 2 * pi, num=precision)

    for z in z_angles[1:]:

        direction = start_dir.copy()
        rotation = Euler((0, 0, z))
        direction.rotate(rotation)

        rotation_axis = Vector((0, 1, 0))
        plane_normal = rotation_axis
        x_vec = direction
        y_vec = x_vec.cross(plane_normal).normalized()

        angles = np.linspace(0, 2 * pi, num=precision)
        for angle in angles:
            destination = x_vec * sin(angle) + y_vec * cos(angle)
            directions.append(destination)

    return directions


def get_max_distance_on_plane(origin, plane_normal):
    locations = raycast_along_plane(origin, plane_normal, n_raycasts=16)
    distances = [(loc - origin).length for loc in locations]
    return max(distances)


def add_camera_mladjo(name="Camera", cam_type="ORTHO"):
    cam = bpy.data.cameras.new(name)
    cam.type = cam_type
    if cam_type == "ORTHO":
        cam.ortho_scale = 1.0
    cam_obj = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(cam_obj)
    return cam_obj


def center_along_axis_with_max_ratio(origin, locations):
    half = int(len(locations) / 2)
    locs_tup = [(locations[i], locations[i + half]) for i in range(half)]
    max_ratio = -1000
    max_ratio_index = -1
    for (j, loc) in enumerate(locs_tup):
        dist_1 = (loc[0] - origin).length
        dist_2 = (loc[1] - origin).length
        ratio = max(dist_1, dist_2) / min(dist_1, dist_2)
        if ratio > max_ratio:
            max_ratio = ratio
            max_ratio_index = j
    center = (locs_tup[max_ratio_index][0] + locs_tup[max_ratio_index][1]) / 2
    return center


def make_mesh_transparent(mesh):
    material = mesh.data.materials[0]
    material.blend_method = 'BLEND'
    node = material.node_tree.nodes["Principled BSDF"]
    node.inputs['Alpha'].default_value = 0.15


def set_emission(mesh):
    if len(mesh.data.materials) > 0:
        material = mesh.data.materials[0]
    else:
        material = bpy.data.materials.new(name="Material")
        material.use_nodes = True
        mesh.data.materials.append(material)

    node = material.node_tree.nodes["Principled BSDF"]
    node.inputs['Emission'].default_value = (1, 0.000901322, 0, 1)


def rotate_object_around_axis(obj, angle_deg, axis):
    rot_mat = Matrix.Rotation(radians(angle_deg), 4, axis)

    mat_w = obj.matrix_world
    # decompose world_matrix's components, and from them assemble 4x4 matrices
    orig_loc, orig_rot, orig_scale = mat_w.decompose()
    orig_loc_mat = Matrix.Translation(orig_loc)
    orig_rot_mat = orig_rot.to_matrix().to_4x4()
    orig_scale_mat = Matrix.Scale(orig_scale[0], 4, (1, 0, 0)) @ Matrix.Scale(
                                orig_scale[1], 4, (0, 1, 0)) @ Matrix.Scale(
                                orig_scale[2], 4, (0, 0, 1))

    # assemble the new matrix
    obj.matrix_world = orig_loc_mat @ rot_mat @ orig_rot_mat @ orig_scale_mat


def get_mesh_limits(mesh):
    mat_w = mesh.matrix_world
    x_min = 1000000
    x_max = -1000000
    y_min = 1000000
    y_max = -1000000
    z_min = 1000000
    z_max = -1000000

    for v in mesh.data.vertices:
        co = mat_w @ v.co

        if co.x < x_min:
            x_min = co.x
        if co.x > x_max:
            x_max = co.x

        if co.y < y_min:
            y_min = co.y
        if co.y > y_max:
            y_max = co.y

        if co.z < z_min:
            z_min = co.z
        if co.z > z_max:
            z_max = co.z

    return x_min, x_max, y_min, y_max, z_min, z_max


def get_polys_by_height_increments(mesh, increment):
    n = ceil(mesh.dimensions.z / increment)
    polys_by_height = [None] * n
    for i in range(n):
        polys_by_height[i] = set()

    for poly in mesh.data.polygons:
        indexes = []
        for vInd in poly.vertices:
            vertex_co = mesh.matrix_world @ mesh.data.vertices[vInd].co
            z = vertex_co.z
            try:
                polys_by_height[floor(z/increment)].add(poly)
            except Exception:
                pass
            #    print("FIX HEIGHT OF MODEL")
            indexes.append(floor(z/increment))

        min_index = min(indexes)
        max_index = max(indexes)
        for i in range(min_index, max_index):
            try:
                polys_by_height[i].add(poly)
            except Exception:
                pass

    return polys_by_height


def visit_all_neighbors(poly,
                        polys_dict,
                        visited,
                        not_visited,
                        edges_key_to_polys):
    visited[poly] = True
    not_visited.pop(poly, None)
    neighb = get_neighb_of_poly(poly, edges_key_to_polys)
    for p in neighb:
        if (p not in visited) and (p in polys_dict):
            visit_all_neighbors(p, polys_dict, visited,
                                not_visited, edges_key_to_polys)


def get_islands(mesh, polys, edges_key_to_polys):
    polys_dict = {}
    visited = {}
    not_visited = {}
    islands = []
    centers = []
    for p in polys:
        polys_dict[p] = True
        not_visited[p] = True

    nIslands = 0
    while len(not_visited) > 0:
        visited.clear()
        visit_all_neighbors(next(iter(not_visited.keys())),
                            polys_dict, visited,
                            not_visited, edges_key_to_polys)
        center = Vector((0, 0, 0))
        islands.append([])
        for poly in visited:
            islands[nIslands].append(poly)
            center += poly.center
        center /= len(visited)
        centers.append(center)
        nIslands += 1
    return nIslands, islands, centers


def create_seam(mesh, first_index, scnd_index):
    # function that take mesh, and two vertex indexes and make seam
    start_time = time.time()
    mesh.data.vertices[first_index].select = True
    mesh.data.vertices[scnd_index].select = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.shortest_path_select()
    bpy.ops.mesh.mark_seam(clear=False)
    deselect_all_in_edit(mesh)
    end_time = time.time()
    print("seam made in: ")
    print(end_time-start_time)


def get_vertices_using_raycast_for_arms(centers, mesh):
    # this can beetter, but works, when have time fix it,
    # arms sor using y, other using x
    centers = sorted(centers, key=lambda y: y[1])
    indexes_of_polys = []
    indexes_of_vertices = []
    number_of_center = len(centers)

    for x in range(number_of_center):
        center_0 = centers[x]
        mat = mesh.matrix_world
        # this can faster
        vert = [mat @ v.co for v in mesh.data.vertices]
        poly = [p.vertices for p in mesh.data.polygons]
        bvh = BVHTree.FromPolygons(vert, poly)
        vec0 = Vector((0, 1, 0))
        vec1 = Vector((0, -1, 0))
        vec2 = Vector((1, 0, 0))
        vec3 = Vector((-1, 0, 0))
        vectors = [vec0, vec1, vec2, vec3]
        for vec in vectors:
            loc, norm, index, dist = bvh.ray_cast(center_0, vec)
            if loc:
                indexes_of_polys.append(index)
    for x in indexes_of_polys:
        indexes_of_vertices.append(mesh.data.polygons[x].vertices[0])

    return indexes_of_vertices


def get_vertices_using_raycast(centers, mesh):
    centers = sorted(centers, key=lambda x: x[0])
    indexes_of_polys = []
    indexes_of_vertices = []
    number_of_center = len(centers)
    for x in range(number_of_center):
        center_0 = centers[x]
        mat = mesh.matrix_world
        # faster
        vert = [mat @ v.co for v in mesh.data.vertices]
        poly = [p.vertices for p in mesh.data.polygons]
        bvh = BVHTree.FromPolygons(vert, poly)
        vec0 = Vector((0, 1, 0))
        vec1 = Vector((0, -1, 0))
        vec2 = Vector((1, 0, 0))
        vec3 = Vector((-1, 0, 0))
        vectors = [vec0, vec1, vec2, vec3]
        for vec in vectors:
            loc, norm, index, dist = bvh.ray_cast(center_0, vec)
            if loc:
                indexes_of_polys.append(index)

    for x in indexes_of_polys:
        indexes_of_vertices.append(mesh.data.polygons[x].vertices[0])

    return indexes_of_vertices


def get_closest_vert_to_loc(mesh, location):
    list_off_all_distances_1 = []
    for v in mesh.data.vertices:
        final_co = mesh.matrix_world @ v.co
        p1 = np.array([final_co[0], final_co[1], final_co[2]])
        p2 = np.array([location[0], location[1], location[2]])
        squared_dist = np.sum((p1-p2)**2, axis=0)
        dist = np.sqrt(squared_dist)
        list_off_all_distances_1.append(dist)

    min_dist_1 = min(list_off_all_distances_1)

    for v in mesh.data.vertices:
        final_co = mesh.matrix_world @ v.co
        p1 = np.array([final_co[0], final_co[1], final_co[2]])
        p2 = np.array([location[0], location[1], location[2]])
        squared_dist = np.sum((p1-p2)**2, axis=0)
        dist = np.sqrt(squared_dist)
        if (dist == min_dist_1):
            closest_1 = v
    index_closest_one = closest_1.index
    return index_closest_one
