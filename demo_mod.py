import cv2
import numpy as np
import fbrs_predict
import os
import open3d as o3d
from utils import ProjectToImage, pinhole_model
import json
import pyvista
from scipy.spatial.transform import Rotation

from skspatial.objects import Plane
from skspatial.objects import Point
from skspatial.objects import Vector


def find_mask(image_path):
    image_file = image_path
    downscale = 1  # performance increases with downscaling image (removes high frequencies)

    image = cv2.imread(image_file)
    image = cv2.resize(image, (int(image.shape[1] * downscale), int(image.shape[0] * downscale)))
    os.path.splitext(image_file)[0]
    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    checkpoint = 'resnet34_dh128_sbd'  # download a pretrained model from https://github.com/cviss-lab/fbrs_interactive_segmentation to /weights
    engine = fbrs_predict.fbrs_engine(checkpoint)

    x_coord = []
    y_coord = []
    is_pos = []

    def interactive_win(event, u, v, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            x_coord.append(u)
            y_coord.append(v)
            is_pos.append(1)
            cv2.circle(image2, (u, v), int(5), (0, 255, 0), -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            x_coord.append(u)
            y_coord.append(v)
            is_pos.append(0)
            cv2.circle(image2, (u, v), int(5), (255, 0, 0), -1)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', interactive_win)

    image2 = image

    while (1):
        cv2.imshow('image', image2)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # 'Esc' Key
            break

        if k == 13:  # 'Enter' Key

            image = cv2.imread(image_file)
            image = cv2.resize(image, (int(image.shape[1] * downscale), int(image.shape[0] * downscale)))

            mask_pred = engine.predict(x_coord, y_coord, is_pos, image, brs_mode='f-BRS-B')  # F-BRS Prediction Function

            if len(image.shape) == 3:
                mask = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
            else:
                mask = np.zeros((image.shape[0], image.shape[1]))

            alpha = 0.8
            mask[mask_pred != 0, :] = [255, 255, 255]
            image[mask_pred != 0, :] = alpha * mask[mask_pred != 0, :] + (1 - alpha) * image[mask_pred != 0, :]
            image2 = np.array(image, dtype=np.uint8)
            cv2.imshow('mask', image2)
            cv2.waitKey(0)
            cv2.imwrite(os.path.splitext(image_file)[0] + '_mask.jpg', mask)
            return mask


def find_contours(img_path):
    mask = cv2.imread(os.path.splitext(img_path)[0] + '_mask.jpg')
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(
        mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def read_ply(pointcloud):
    ply = o3d.io.read_point_cloud(pointcloud)
    ply_points = np.asarray(ply.points)
    ply_colors = np.asarray(ply.colors)
    ply_normals = np.asarray(ply.normals)

    # plane_model, inliers = ply.segment_plane(distance_threshold=0.01,
    #                                          ransac_n=3,
    #                                          num_iterations=10000)[a, b, c, d] = plane_model
    # plane_ply = ply.select_by_index(inliers)
    # plane_ply.paint_uniform_color([1.0, 0, 0])
    # stockpile_ply = ply.select_by_index(inliers, invert=True)
    # stockpile_ply.paint_uniform_color([0, 0, 1.0])
    # o3d.visualization.draw_geometries([plane_ply, stockpile_ply, axes])

    return ply_points, ply_colors, ply_normals


def read_json(json_file):
    with open(json_file, 'r') as f:
        json_read = json.load(f)

    return json_read


def get_K_from_SFM_json(SFM_json):
    K = np.zeros((3, 3))
    SFM_json['intrinsics'][0]['value']['ptr_wrapper']['data']['focal_length']
    K[0, 0] = SFM_json['intrinsics'][0]['value']['ptr_wrapper']['data']['focal_length']
    K[1, 1] = SFM_json['intrinsics'][0]['value']['ptr_wrapper']['data']['focal_length']

    K[0, 2] = SFM_json['intrinsics'][0]['value']['ptr_wrapper']['data']['principal_point'][0]
    K[1, 2] = SFM_json['intrinsics'][0]['value']['ptr_wrapper']['data']['principal_point'][1]

    K[2, 2] = 1

    return K


def get_P_from_SFM_json(SFM_json, idx, K):
    for i in range(len(SFM_json['extrinsics'])):

        if SFM_json['views'][i]['value']['ptr_wrapper']['data']['filename'] == str(idx) + '.jpg':
            req_key = SFM_json['views'][i]['key']

            req_rot = np.array(SFM_json['extrinsics'][req_key]['value']['rotation'])
            req_c = np.array(SFM_json['extrinsics'][req_key]['value']['center'])
            req_c = np.array(req_c).reshape(-1, 1)

            R = req_rot
            t = R.dot(-req_c)

            # Construct projection matrix (P)
            P = K @ np.hstack([R, t])

    return P, R, req_c


# https://www.programcreek.com/python/?CodeExample=fit+plane
def fit_plane_to_points(points, return_meta=False):
    """Fit a plane to a set of points.

    Parameters
    ----------
    points : np.ndarray
        Size n by 3 array of points to fit a plane through

    return_meta : bool
        If true, also returns the center and normal used to generate the plane

    """
    data = np.array(points)
    center = data.mean(axis=0)
    result = np.linalg.svd(data - center)
    normal = np.cross(result[2][0], result[2][1])
    plane = pyvista.Plane(center=center, direction=normal)
    if return_meta:
        return plane, center, normal
    return plane


def ProjectToImageThis(projectionMatrix, pos):
    """Project 3D world coordinates to 2D image coordinates using a pinhole camera model

    Arguments
    -------
        P (numpy.array): projection matrix
        pos (numpy.array): 3D world coordinates (3xN)

    Returns
    -------
        uv (numpy.array): 2D pixel coordinates (2xN)
    """
    pos = np.array(pos).reshape(3, -1)
    pos_ = np.vstack([pos, np.ones((1, pos.shape[1]))])

    uv_ = np.dot(projectionMatrix, pos_)
    uv = uv_[:-1, :] / uv_[-1, :]

    return uv


def saveProjectImage(img, pts_3d, path):
    for points in pts_3d:
        img_project = cv2.circle(img, (int(points[0]), int(points[1])), radius=1, thickness=1,
                                 color=(255, 255, 255))
    cv2.imwrite(path + '_project.jpg', img_project)


def checkMask(binary_mask, pts_3d):
    inner_index = []
    outer_index = []

    mask_h = binary_mask.shape[0]
    mask_w = binary_mask.shape[1]

    for count, points in enumerate(pts_3d):
        if (0 < points[0] < mask_w) and (0 < points[1] < mask_h):
            if binary_mask[int(points[1]), int(points[0])] >= 250:
                inner_index.append(count)
            else:
                outer_index.append(count)

    inner_index = np.array(inner_index)
    outer_index = np.array(outer_index)

    return inner_index, outer_index


def writePly(file, index, path, name):
    pcd = o3d.geometry.PointCloud()

    main_pcd = o3d.io.read_point_cloud(file)
    main_pcd_points = np.asarray(main_pcd.points)
    main_pcd_colors = np.asarray(main_pcd.colors)

    pcd.colors = o3d.utility.Vector3dVector(main_pcd_colors[index])
    pcd.points = o3d.utility.Vector3dVector(main_pcd_points[index])
    o3d.io.write_point_cloud(path + name + ".ply", pcd)


def createMesh(points, colors, path, name):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals()
    poisson_mesh = \
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1,
                                                                  linear_fit=False)[0]
    o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)
    o3d.io.write_triangle_mesh(path + name + ".ply", p_mesh_crop)
    return p_mesh_crop


if __name__ == '__main__':

    # inputs
    img_index = 65
    image_dir_path = '/home/jp/Desktop/Rishabh/HL_2_Data/defect_2_fbrs/images/'
    ply_file = '/home/jp/Desktop/Rishabh/HL_2_Data/defect_2_fbrs/openMVG_ULTRA/reconstruction_global/UNKNOWN_MVG_colorized.ply'
    data_dir = '/home/jp/Desktop/Rishabh/HL_2_Data/defect_2_fbrs/'

    image_path = image_dir_path + str(img_index) + '.jpg'
    output_path = os.path.splitext(image_path)[0]
    mask_path = output_path + '_mask.jpg'

    print("Line 262")

    image = cv2.imread(image_path)

    # segment spalling using fbrs
    if not os.path.exists(mask_path):
        mask = find_mask(image_path)
        mask_simple = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if os.path.exists(mask_path):
        mask_simple = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print("Here")
    contour = find_contours(image_path)

    # read the sfm pointcloud file
    pcd_points, pcd_colors, pcd_normals = read_ply(ply_file)


    gt_pcd_points, gt_pcd_colors, gt_pcd_normals = read_ply("/home/jp/Desktop/Rishabh/Volumetric_Segmentation/Ground_truths/21-06-06 gardiner intel realsense/ply/defect_2_spall.ply")


    # read intrinsics and projection matrix
    sfm_file_unknown = data_dir + 'openMVG_ULTRA/reconstruction_global/sfm_data.json'
    sfm_unknown = read_json(sfm_file_unknown)

    K_unknown = get_K_from_SFM_json(sfm_unknown)

    P, _, _ = get_P_from_SFM_json(sfm_unknown, img_index, K_unknown)

    # project the 3d points to the chosen image
    UV = ProjectToImageThis(P, pcd_points.T).T
    # clean up
    UV_defined = UV[UV < np.inf].reshape((-1, 2))
    # save image with projected points
    saveProjectImage(image, UV_defined, output_path)

    # find the indices of inner points and outer points
    in_points_idx, out_points_idx = checkMask(mask_simple, UV_defined)

    # find the inner/outer 3d points from indices
    in_points_3d = pcd_points[in_points_idx, :]
    out_points_3d = pcd_points[out_points_idx, :]

    # save inner and outer pointclouds
    writePly(ply_file, in_points_idx, output_path, "inner_points")
    writePly(ply_file, out_points_idx, output_path, "outer_points")

    # create the mesh for inner points
    mesh = createMesh(in_points_3d, pcd_colors[in_points_idx], output_path, "mesh_cropped")
    gt_mesh = createMesh(gt_pcd_points,gt_pcd_colors, output_path, "defect_2_gt")
    # delete the vertices and triangles outside the mask
    vertices = np.asarray(mesh.vertices)
    vertices_2d = ProjectToImageThis(P, vertices.T).T
    in_vertices_idx, out_vertices_idx = checkMask(mask_simple, vertices_2d)
    # vertices[out_vertices_idx, :] = np.Inf
    # vertices_defined = vertices[vertices < np.inf].reshape((-1, 3))

    vertices_mask = np.zeros((len(vertices_2d)))
    vertices_mask[out_vertices_idx] = 1
    o3d.geometry.TriangleMesh.remove_vertices_by_mask(mesh, vertices_mask)
    o3d.io.write_triangle_mesh(output_path + "cleaned_mesh.ply", mesh)

    # to fit a plane, do a dilation and subtract original mask to get the boundary of the spall

    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((10, 10), np.uint8)

    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    mask_simple_dilation = cv2.dilate(mask_simple, kernel, iterations=1) - mask_simple

    in_plane_points_idx, out_plane_points_idx = checkMask(mask_simple_dilation, UV_defined)

    # find the inner/outer 3d points from indices
    in_plane_points_3d = pcd_points[in_plane_points_idx, :]
    out_plane_points_3d = pcd_points[out_plane_points_idx, :]

    # save inner and outer pointclouds
    writePly(ply_file, in_plane_points_idx, output_path, "inner_plane_points")
    writePly(ply_file, out_plane_points_idx, output_path, "outer_plane_points")

    plane = createMesh(in_plane_points_3d, pcd_colors[in_plane_points_idx], output_path, "plane_mesh")

    # print(mesh.get_volume())
    print(mesh.is_watertight())

    # print(plane.get_volume())
    print(plane.is_watertight())

    plane_pcd = o3d.io.read_point_cloud(
        "/home/jp/Desktop/Rishabh/HL_2_Data/defect_2_fbrs/images/65inner_plane_points.ply")
    # plane_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16),
    #                      fast_normal_computation=True)

    ransac_thresh = 0.001
    plane_model, inliers = plane_pcd.segment_plane(distance_threshold=ransac_thresh, ransac_n=3, num_iterations=1000)

    pcd_points, pcd_colors, pcd_normals = read_ply(
        "/home/jp/Desktop/Rishabh/HL_2_Data/defect_2_fbrs/images/65inner_plane_points.ply")
    inlier_plane_3d = pcd_points[inliers, :]
    inlier_plane = createMesh(inlier_plane_3d, pcd_colors[inliers], output_path,
                              str(ransac_thresh) + "_inlier_plane_mesh")

    for points in inlier_plane_3d:
        zeros2 = plane_model[0] * points[0] + plane_model[1] * points[1] + plane_model[2] * points[2] + plane_model[3]

    plane = Plane(point=[points[0], points[1], points[2]], normal=[plane_model[0], plane_model[1], plane_model[2]])


    def unit_normal(a, b, c):
        x = np.linalg.det([[1, a[1], a[2]],
                           [1, b[1], b[2]],
                           [1, c[1], c[2]]])
        y = np.linalg.det([[a[0], 1, a[2]],
                           [b[0], 1, b[2]],
                           [c[0], 1, c[2]]])
        z = np.linalg.det([[a[0], a[1], 1],
                           [b[0], b[1], 1],
                           [c[0], c[1], 1]])
        magnitude = (x ** 2 + y ** 2 + z ** 2) ** .5
        return x / magnitude, y / magnitude, z / magnitude


    # area of polygon poly
    def poly_area(poly):
        if len(poly) < 3:  # not a plane - no area
            return 0
        total = [0, 0, 0]
        N = len(poly)
        for i in range(N):
            vi1 = poly[i]
            vi2 = poly[(i + 1) % N]
            prod = np.cross(vi1, vi2)
            total[0] += prod[0]
            total[1] += prod[1]
            total[2] += prod[2]
        result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
        return abs(result / 2)


    areas = []
    volumes = []
    dist_list = []
    projected_points = []
    plane_sides = []
    checks = []
    for i in range(len(mesh.triangles)):

        polygon = []
        dist = 0
        poly = []
        check = 1
        for j in range(3):

            point = Point(np.array(mesh.vertices[mesh.triangles[i][j]]))
            plane_side = plane_model[0] * point[0] + plane_model[1] * point[1] + plane_model[2] * point[2] + plane_model[3]
            if plane_side<0:
                check = 0

            plane_sides.append(plane_side)
            point_projected = plane.project_point(point)
            temp = np.linalg.norm(point - point_projected)
            dist += np.linalg.norm(point - point_projected)
            poly.append(point_projected)
            projected_points.append(point_projected)
        avg_dist = dist / 3
        avg_dist = avg_dist * check
        dist_list.append(dist)
        checks.append(check)
        area = poly_area(poly)*check
        volume = area * avg_dist
        areas.append(area)
        volumes.append(volume)

    total_vol = sum(volumes)

    print(total_vol)
