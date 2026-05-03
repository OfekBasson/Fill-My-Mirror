import bpy
import sys
import numpy as np
from mathutils import Matrix


CAMERA_MATRIX_WORLD = Matrix((
    (-1.0, 6.6002361555513294e-15, -8.742277657347586e-08, 0.0),
    (-8.742277657347586e-08, -7.549790126404332e-08, 1.0, 0.0),
    (0.0, 1.0, 7.549790126404332e-08, 0.0),
    (0.0, 0.0, 0.0, 1.0)
))


def setup_textured_materials():
    scene = bpy.context.scene
    
    for obj in scene.objects:
        if obj.type != "MESH":
            continue

        for slot in obj.material_slots:
            mat = slot.material
            if not mat or not mat.use_nodes:
                continue

            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            existing_image = None
            for node in nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    existing_image = node.image
                    break

            if existing_image is None:
                continue

            nodes.clear()

            tex_node = nodes.new(type="ShaderNodeTexImage")
            tex_node.image = existing_image
            tex_node.interpolation = "Linear"
            tex_node.image.colorspace_settings.name = 'AgX Base sRGB'

            emission_front = nodes.new(type="ShaderNodeEmission")
            emission_back = nodes.new(type="ShaderNodeEmission")
            emission_back.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            emission_back.inputs["Strength"].default_value = 1.0
            mat.use_backface_culling = False

            geometry_node = nodes.new(type="ShaderNodeNewGeometry")
            mix_shader = nodes.new(type="ShaderNodeMixShader")
            output_node = nodes.new(type="ShaderNodeOutputMaterial")

            mat.use_backface_culling = False

            links.new(tex_node.outputs["Color"], emission_front.inputs["Color"])
            links.new(geometry_node.outputs["Backfacing"], mix_shader.inputs["Fac"])

            # front-facing = textured, back-facing = black
            links.new(emission_front.outputs["Emission"], mix_shader.inputs[2])
            links.new(emission_back.outputs["Emission"], mix_shader.inputs[1])
            links.new(mix_shader.outputs["Shader"], output_node.inputs["Surface"])


def setup_bw_materials():
    scene = bpy.context.scene

    if not bpy.data.worlds:
        world = bpy.data.worlds.new("World")
        scene.world = world
    else:
        world = bpy.data.worlds[0]
        scene.world = world

    world.use_nodes = True
    bg_nodes = world.node_tree.nodes
    bg_links = world.node_tree.links
    bg_nodes.clear()

    bg_output = bg_nodes.new(type="ShaderNodeOutputWorld")
    bg_background = bg_nodes.new(type="ShaderNodeBackground")
    bg_background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_links.new(bg_background.outputs["Background"], bg_output.inputs["Surface"])

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue

        mat = bpy.data.materials.new(name=f"{obj.name}_BW")
        mat.use_nodes = True
        mat.use_backface_culling = False

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        geo_node = nodes.new(type="ShaderNodeNewGeometry")
        emission_black = nodes.new(type="ShaderNodeEmission")
        emission_white = nodes.new(type="ShaderNodeEmission")
        mix_shader = nodes.new(type="ShaderNodeMixShader")
        output_node = nodes.new(type="ShaderNodeOutputMaterial")

        emission_black.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        emission_white.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

        links.new(geo_node.outputs["Backfacing"], mix_shader.inputs["Fac"])

        links.new(emission_black.outputs["Emission"], mix_shader.inputs[2])
        links.new(emission_white.outputs["Emission"], mix_shader.inputs[1])
        links.new(mix_shader.outputs["Shader"], output_node.inputs["Surface"])

        obj.data.materials.clear()
        obj.data.materials.append(mat)


def setup_camera(intrinsics, height, width):
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    fx_norm = intrinsics[0, 0]

    sensor_width = cam.data.sensor_width
    fx_px = fx_norm * width
    focal_length_mm = fx_px * (sensor_width / width)

    cam.data.lens = focal_length_mm

    cam.matrix_world = CAMERA_MATRIX_WORLD


def setup_depth_output(scene, depth_path: str):
    scene.use_nodes = True
    scene.view_layers[0].use_pass_z = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.base_path = ""
    fo.file_slots[0].path = depth_path
    fo.format.file_format = "OPEN_EXR"
    fo.format.color_depth = "32"
    tree.links.new(rl.outputs["Depth"], fo.inputs[0])


def main():
    argv = sys.argv
    args = argv[argv.index("--") + 1:]

    glb_path = args[0]
    output_path = args[1]
    bw_output_path = args[2]
    npz_path = args[3]
    depth_output_path = args[4] if len(args) > 4 else None

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.use_preferences_save = False

    data = np.load(npz_path, allow_pickle=True)
    intrinsics = data["intrinsics"]
    image_shape = data["image_shape"]

    height = int(image_shape[0])
    width = int(image_shape[1])

    bpy.ops.import_scene.gltf(filepath=glb_path)

    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    setup_camera(intrinsics, height, width)

    setup_textured_materials()
    if depth_output_path is not None:
        setup_depth_output(scene, depth_output_path)
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

    # Disable compositor so B&W render doesn't re-trigger depth output
    if depth_output_path is not None:
        scene.use_nodes = False

    setup_bw_materials()
    scene.render.filepath = bw_output_path
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()