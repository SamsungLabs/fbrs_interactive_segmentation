import os
import json
import numpy as np
from scipy.optimize import leastsq
from scipy.spatial.transform import Rotation

def pinhole_model(pose, K):
    """Loads projection and extrensics information for a pinhole camera model

    Arguments
    -------
        pose (numpy.array): px, py, cx, qx, qy, qz, qw
        K (numpy.array): camera matrix

    Returns
    -------
        P (numpy.array): projection matrix
        R (numpy.array): rotation matrix (camera coordinates)
        C (numpy.array): camera center (world coordinates)
    """

    pose = pose.reshape(-1)
    C = np.array(pose[1:4]).reshape(-1, 1)

    # Convert camera rotation from quaternion to matrix
    q = pose[4:]
    r = Rotation.from_quat(q)
    Rot = r.as_matrix()

    # Find camera extrinsics (R,t)
    R = Rot.T
    t = Rot.T.dot(-C)

    # Construct projection matrix (P)
    P = K @ np.hstack([R, t])

    return P, R, C


def ProjectToImage(projectionMatrix, pos):
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
    # uv_ = uv_[:, uv_[-1, :] > 0]
    uv = uv_[:-1, :] / uv_[-1, :]

    return uv


def ProjectToWorld(projectionMatrix, uv, R, C):
    """Back-project 2D image coordinates to rays in 3D world coordinates using a pinhole camera model

    Arguments
    -------
        P (numpy.array): projection matrix
        uv (numpy.array): 2D pixel coordinates (2xN)
        R (numpy.array): rotation matrix (camera coordinates)
        C (numpy.array): camera center (world coordinates)

    Returns
    -------
        pos (numpy.array): [3D world coordinates (3xN)]
    """
    uv_ = np.vstack([uv[0, :], uv[1, :], np.ones((1, uv.shape[1]))])
    pinvProjectionMatrix = np.linalg.pinv(projectionMatrix)

    pos2_ = np.dot(pinvProjectionMatrix, uv_)
    pos2_[-1, pos2_[-1, :] == 0] = 1
    pos2 = pos2_[:-1, :] / pos2_[-1, :]
    rays = pos2 - C

    # check that rays project forwards
    rays_local = R @ rays
    rays[:, rays_local[2, :] < 0] = -1 * rays[:, rays_local[2, :] < 0]
    rays = rays / np.linalg.norm(rays, axis=0)

    return rays


def test1():
    max_error = 0
    data_dir = '/home/jp/Desktop/Rishabh/HL_2_Data/defect_2/'
    poses_file = os.path.join(data_dir, 'poses.csv')
    intrinsics_file = os.path.join(data_dir, 'intrinsics.json')

    with open(intrinsics_file, 'r') as f:
        intrinsics = json.load(f)
    K = np.array(intrinsics['camera_matrix'])
    H = np.array(intrinsics['height'])
    W = np.array(intrinsics['width'])
    cx = K[0, 2]
    cy = K[1, 2]

    poses = np.loadtxt(poses_file, delimiter=",")

    for pose in poses:
        P, R, C = pinhole_model(pose, K)
        uv = np.array([[0, 0], [W, H], [cx, cy]]).T
        rays = ProjectToWorld(P, uv, R, C)

        # test functions by calculating reprojection error
        p2 = C + 1 * rays
        uv_reprojected = ProjectToImage(P, p2)
        error = np.sum((uv - uv_reprojected)[0, :] ** 2 + (uv - uv_reprojected)[1, :] ** 2)
        if error > max_error:
            max_error = error

    print('Test 1: Maximum Reprojection Error = %e' % max_error)


def test2():
    import cv2

    data_dir = '/home/jp/Desktop/Rishabh/HL_2_Data/defect_2/'
    poses_file = os.path.join(data_dir, 'poses.csv')
    intrinsics_file = os.path.join(data_dir, 'intrinsics.json')

    with open(intrinsics_file, 'r') as f:
        intrinsics = json.load(f)
    K = np.array(intrinsics['camera_matrix'])
    poses = np.loadtxt(poses_file, delimiter=",")

    index1 = 33
    index2 = 37
    pose1 = poses[index1]
    pose2 = poses[index2]

    i1 = int(pose1[0])
    I1 = cv2.imread(os.path.join(data_dir, 'images', str(i1) + '.jpg'))
    P1, _, _ = pinhole_model(pose1, K)

    i2 = int(pose2[0])
    I2 = cv2.imread(os.path.join(data_dir, 'images', str(i2) + '.jpg'))
    P2, _, _ = pinhole_model(pose2, K)

    # ORB features matching
    orb = cv2.ORB_create()

    # Find keypoints and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY), None)
    keypoints2, descriptors2 = orb.detectAndCompute(cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY), None)

    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:200]

    ORB_matches = cv2.drawMatches(I1, keypoints1, I2, keypoints2, matches, None, flags=2)
    ORB_matches = cv2.resize(ORB_matches, (1280, 720))
    cv2.imshow('', ORB_matches)

    uv1 = np.array([keypoints1[m.queryIdx].pt for m in matches]).T
    uv2 = np.array([keypoints2[m.trainIdx].pt for m in matches]).T

    points4d = cv2.triangulatePoints(P1, P2, uv1, uv2)
    points3d = (points4d[:3, :] / points4d[3, :])

    uv1_reprojected = ProjectToImage(P1, points3d)
    uv2_reprojected = ProjectToImage(P2, points3d)
    error1 = np.sum((uv1 - uv1_reprojected)[0, :] ** 2 + (uv1 - uv1_reprojected)[1, :] ** 2)
    error2 = np.sum((uv2 - uv2_reprojected)[0, :] ** 2 + (uv2 - uv2_reprojected)[1, :] ** 2)
    max_error = np.max([error1, error2])

    print('Test 2: Maximum Reprojection Error = %e' % max_error)

    x = points3d[0, :]
    y = points3d[1, :]
    z = points3d[2, :]

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import proj3d
    #
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.scatter(x, y, z)
    # # ax.set_xlim3d(np.ptp(x))
    # # ax.set_ylim3d(np.ptp(y))
    # ax.set_zlim3d(np.ptp(z))
    # plt.show()


def plane_error(p, xs, ys, zs):
    A = p[0]
    B = p[1]
    C = p[2]
    D = p[3]
    return abs(A * xs + B * ys + C * zs + D) / np.sqrt(A ** 2 + B ** 2 + C ** 2)


def plane3point(x, y, z):
    N = len(x)
    L = np.vstack([x, y, z]).T
    d = np.ones((N, 1))
    A, B, C = np.linalg.solve(L, d)
    p = np.sum([x, y, z], axis=0)
    D = -1 * (A * p[0] + B * p[1] * C * p[2])
    return A, B, C, D


def VectorProject(a, b):
    b_norm = np.linalg.norm(b)
    return a.dot(b) * b / b_norm ** 2


def VectorProjectNorm(a, b):
    b_norm = np.linalg.norm(b)
    return a.dot(b) / b_norm


def plane_leastsq(xs, ys, zs):
    N = len(xs)
    if N < 3:
        return None

    if N == 3:
        xs = np.insert(xs, 0, np.mean(xs))
        ys = np.insert(ys, 0, np.mean(ys))
        zs = np.insert(zs, 0, np.mean(zs))

    p0 = [1, 1, 1, 1]
    sol = leastsq(plane_error, p0, args=(xs, ys, zs))[0]

    return sol


def PlanePointIntersect(origin, ray, A, B, C, D):
    n = np.array([A, B, C]) / np.linalg.norm([A, B, C])
    p0 = np.array([0, 0, -D / C])
    d = np.dot(p0 - origin, n) / np.dot(ray, n)
    return origin + d * ray


def ransac_plane(points, inlier_ratio, max_dist, max_iteration):
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    n_data = points.shape[0]
    n_inlier = 0
    iteration = 0
    ratio = []

    while iteration < max_iteration and n_inlier / n_data < inlier_ratio:
        dist_it = []
        ind = [np.random.randint(0, n_data) for _ in range(int(n_data / 5))]
        xi = np.array([xs[i] for i in ind])
        yi = np.array([ys[i] for i in ind])
        zi = np.array([zs[i] for i in ind])
        try:
            # P = plane3point(xi,yi,zi)
            P = plane_leastsq(xi, yi, zi)
        except:
            continue
        dist_it = plane_error(P, xs, ys, zs)
        index_inliers = np.where(dist_it <= max_dist)
        n_inlier = sum(dist_it <= max_dist)
        ratio.append(n_inlier / n_data)
        iteration += 1
    A, B, C, D = P
    r = ratio[-1]
    return A, B, C, D, r, iteration, index_inliers


def project_depth_to_xyz(uv, K, D, center=False, h=0):
    v = uv[1, :]
    u = uv[0, :]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    d = []

    if center:
        d0 = D[int(v.mean()), int(u.mean())]
        d = [d0 for _ in range(u.shape[0])]
    else:
        for ui, vi in zip(u, v):
            di = D[int(vi) - h:int(vi) + h + 1, int(ui) - h:int(ui) + h + 1]
            if len(di) > 0:
                di = di[di > 0].mean()
                d.append(di)

    Z = np.array(d, dtype=np.float) / 1000
    X = Z * (u - cx) / fx
    Y = Z * (v - cy) / fy

    return X, Y, Z


if __name__ == '__main__':
    test1()
    test2()
