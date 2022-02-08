from pyntcloud import PyntCloud
import os
img_index = 65
image_dir_path = '/home/jp/Desktop/Rishabh/HL_2_Data/defect_2_fbrs/images/'
ply_file = '/home/jp/Desktop/Rishabh/HL_2_Data/defect_2_fbrs/openMVG_ULTRA/reconstruction_global/UNKNOWN_MVG_colorized.ply'
data_dir = '/home/jp/Desktop/Rishabh/HL_2_Data/defect_2_fbrs/'

image_path = image_dir_path + str(img_index) + '.jpg'
output_path = os.path.splitext(image_path)[0]
# tin.LoadFile(output_path + "cleaned_mesh.ply")
# tin.load_array(v, f) # or read arrays from memory
infile=output_path + "cleaned_mesh.ply"
outfile=output_path + "watertight_cleaned_mesh.ply"

diamond = PyntCloud.from_file(infile)
convex_hull_id = diamond.add_structure("convex_hull")
convex_hull = diamond.structures[convex_hull_id]
diamond.mesh = convex_hull.get_mesh()
diamond.to_file("diamond_hull.ply", also_save=["mesh"])
volume = convex_hull.volume