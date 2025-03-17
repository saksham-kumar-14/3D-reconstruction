"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import cv2 as cv
import helper as hlp
from scipy.signal import convolve2d

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def normalize_pts(pts, M):
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0)
    T = np.array([
        [1/std[0], 0, -mean[0]/std[0]],
        [0, 1/std[1], -mean[1]/std[1]],
        [0, 0, 1]
    ])
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T

def eight_point(pts1, pts2, M):

    assert M >= 8
    N = pts1.shape[0]

    # Normalize Points
    norm_pts1, T1 = normalize_pts(pts1, M)
    norm_pts2, T2 = normalize_pts(pts2, M)

    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = norm_pts1[i]
        x2, y2 = norm_pts2[i]
        A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

    # SVD implementation
    U, S, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)

    # Enforce rank 2 constraint
    U, S, Vt = np.linalg.svd(F_norm)
    S[2] = 0
    F_rank2 = U @ np.diag(S) @ Vt

    # refining F
    F_refined = hlp.refineF(F_rank2, norm_pts1, norm_pts2)

    # Unscaling Fundamental Matrix
    F = T2.T @ F_refined @ T1

    # Normalize F so that F[2,2] = 1 (convention to makes less abitrary values)
    F /= F[2, 2]

    return F


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def compute_epipolar_lines(F, pts1):
    new_pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    lines = (F @ new_pts1.T).T
    return lines

def epipolar_correspondences(im1, im2, F, pts1):

    window_size = 10

    pts2 = []
    half_w = window_size // 2
    im1_gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2_gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    
    for pt in pts1:
        x, y = int(pt[0]), int(pt[1])
        
        # Compute the epipolar line in the second image
        epipolar_line = F @ np.array([x, y, 1])  # l' = F * x
        a, b, c = epipolar_line
        
        # Generate candidate points along the epipolar line in image 2
        height, width = im2_gray.shape
        candidates = []
        for x2 in range(max(0, x - 50), min(width, x + 50)):
            y2 = int((-a * x2 - c) / b) if abs(b) > 1e-6 else y 
            if 0 <= y2 < height:
                candidates.append((x2, y2))
        
        # Extract window around (x, y) in image 1
        if y - half_w < 0 or y + half_w >= height or x - half_w < 0 or x + half_w >= width:
            pts2.append((x, y))
            continue
        patch1 = im1_gray[y-half_w:y+half_w+1, x-half_w:x+half_w+1]
        
        best_match = None
        min_ncc = -1  # For NCC, higher is better
        
        for (x2, y2) in candidates:
            if y2 - half_w < 0 or y2 + half_w >= height or x2 - half_w < 0 or x2 + half_w >= width:
                continue
            patch2 = im2_gray[y2-half_w:y2+half_w+1, x2-half_w:x2+half_w+1]
            
            # Compute Normalized Cross-Correlation (NCC) for better accuracy
            patch1_norm = (patch1 - np.mean(patch1)) / (np.std(patch1) + 1e-6)
            patch2_norm = (patch2 - np.mean(patch2)) / (np.std(patch2) + 1e-6)
            ncc = np.sum(patch1_norm * patch2_norm) / (window_size ** 2)
            
            if ncc > min_ncc:
                min_ncc = ncc
                best_match = (x2, y2)
        
        if best_match:
            pts2.append(best_match)
        else:
            pts2.append((x, y))  # Fallback to original point if no match found
    
    return np.array(pts2)

"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    E = K2.T @ F @ K1
    U, S, Vt = np.linalg.svd(E)

    S[2] = 0  # Enforce rank 2
    E = U @ np.diag(S) @ Vt
    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    N = pts1.shape[0]
    pts3D_homogeneous = np.zeros((N, 4))

    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        # Construct A matrix for AX = 0
        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]  # Solution

        # Normalize to make last coordinate 1
        pts3D_homogeneous[i] = X / X[3]

    return pts3D_homogeneous


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):

    # Calculating Optical Centers
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)

    # Calculate new rotation matrix
    r1 = (c1 - c2) / np.linalg.norm(c1 - c2) 
    r2 = np.cross(R1[2, :], r1) / np.linalg.norm(np.cross(R1[2, :], r1))
    r3 = np.cross(r2, r1)

    new_rotation_matrix = np.vstack((r1, r2, r3))
    R1p = new_rotation_matrix
    R2p = new_rotation_matrix

    K1p = K2
    K2p = K2

    # calculate new translation vectors
    t1p = -new_rotation_matrix @ c1
    t2p = -new_rotation_matrix @ c2

    # Calculate rectification matrices
    M1 = (K1p @ R1p) @ np.linalg.inv(K1 @ R1)
    M2 = (K2p @ R2p) @ np.linalg.inv(K2 @ R2)

    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):

    h, w = im1.shape
    dispM = np.zeros((h, w), dtype=np.float32)
    half_w = win_size // 2

    # padding images to handle borders
    im1_padded = np.pad(im1, half_w, mode='constant', constant_values=0)
    im2_padded = np.pad(im2, half_w, mode='constant', constant_values=0)

    min_ssd = np.full((h, w), np.inf)  # Initialize min SSD as large values

    for d in range(max_disp + 1):

        # Shift images to left by d pixels
        shifted_im2 = np.roll(im2_padded, -d, axis=1)

        # squared difference
        ssd = (im1_padded - shifted_im2) ** 2

        kernel = np.ones((win_size, win_size))
        ssd_windowed = convolve2d(ssd, kernel, mode='valid')

        # Update disparity map where SSD is lower
        mask = ssd_windowed < min_ssd
        dispM[mask] = d
        min_ssd[mask] = ssd_windowed[mask]

    return dispM

"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    
    # Compute baseline (b) as the Euclidean distance between optical centers
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)  # Optical center of left camera
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)  # Optical center of right camera
    baseline = np.linalg.norm(c1 - c2)

    # Compute focal length (f) from the left camera's intrinsic matrix
    focal_length = K1[0, 0]  # f = K1(1,1)

    # Avoid division by zero by setting zero disparity values to a small epsilon
    dispM_safe = np.where(dispM > 0, dispM, np.inf)

    # Compute depth map
    depthM = (baseline * focal_length) / dispM_safe

    # Set depth to zero where disparity was zero
    depthM[dispM == 0] = 0

    return depthM


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
