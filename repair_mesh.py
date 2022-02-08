import pymeshfix
import os
import pyvista as pv
tin = pymeshfix.PyTMesh()
img_index = 65
image_dir_path = '/home/jp/Desktop/Rishabh/HL_2_Data/defect_2_fbrs/images/'
ply_file = '/home/jp/Desktop/Rishabh/HL_2_Data/defect_2_fbrs/openMVG_ULTRA/reconstruction_global/UNKNOWN_MVG_colorized.ply'
data_dir = '/home/jp/Desktop/Rishabh/HL_2_Data/defect_2_fbrs/'

image_path = image_dir_path + str(img_index) + '.jpg'
output_path = os.path.splitext(image_path)[0]
# tin.LoadFile(output_path + "cleaned_mesh.ply")
# tin.load_array(v, f) # or read arrays from memory
infile="/media/jp/EMTEC M730/Spall_Defect_2_surface.stl"
outfile="/media/jp/EMTEC M730/Watertight_Spall_Defect_2_surface.stl"
import pymeshfix

mesh=pv.read(infile)
mesh.save("rough_mesh.stl")

# Create TMesh object
tin = pymeshfix.PyTMesh()

tin.load_file("rough_mesh.stl")
# tin.load_array(v, f) # or read arrays from memory

# Attempt to join nearby components
# tin.join_closest_components()

# Fill holes
tin.fill_small_boundaries()
# print('There are {:d} boundaries'.format(tin.boundaries())
#
# # Clean (removes self intersections)
tin.clean(max_iters=1, inner_loops=3)
#
# # Check mesh for holes again
print('There are {:d} boundaries'.format(tin.boundaries()))
#
# # Clean again if necessary...
#
# # Output mesh
tin.save_file(outfile)
#
#  # or return numpy arrays
# vclean, fclean = tin.return_arrays()