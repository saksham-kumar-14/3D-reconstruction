import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt

# 1. Load the two temple images and the points from data/some_corresp.npz
corresp_data = np.load("../data/some_corresp.npz")

pts1 = corresp_data["pts1"]
pts2 = corresp_data["pts2"]
im1 = io.imread("../data/im1.png")
im2 = io.imread("../data/im2.png")

# 2. Run eight_point to compute F

M = max(pts1.shape)
F = sub.eight_point(pts1, pts2, M)
# print("The fundamental matrix computer is: ", F)
# hlp.displayEpipolarF(im1, im2, F)

# 3. Load points in image 1 from data/temple_coords.npz
temple_coord_data = np.load("../data/temple_coords.npz")
temple_coord_pts1 = temple_coord_data["pts1"]


# 4. Run epipolar_correspondences to get points in image 2
temple_coord_pts2 = sub.epipolar_correspondences(im1, im2, F, temple_coord_pts1)
# hlp.epipolarMatchGUI(im1, im2, F)

# 5. Compute the camera projection matrix P1

intrinsics_data = np.load("../data/intrinsics.npz")
K1 = intrinsics_data['K1']
K2 = intrinsics_data['K2']
E = sub.essential_matrix(F, K1, K2)
# print("Calculated essential matrix is : ", E)

# # 6. Use camera2 to get 4 camera projection matrices P2

possible_P2s = hlp.camera2(E)
P2s = []
for i in range(4):
    P2s.append(K2 @ possible_P2s[:, :, i])

# 7. Run triangulate using the projection matrices &&  # 8. Figure out the correct P2
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera 1 Projection with no rotation and scaling

best_P2 = None
best_pts3D = None
max_positive_depth = 0

for P2 in P2s:
    # Run triangulation
    pts3D_homogeneous = sub.triangulate(P1, temple_coord_pts1, P2, temple_coord_pts2)

    # Convert to Cartesian (from homogeneous)
    pts3D = pts3D_homogeneous[:, :3]

    # Compute depth in both cameras
    depth1 = (P1 @ pts3D_homogeneous.T)[2]  # Z-depth in camera 1
    depth2 = (P2 @ pts3D_homogeneous.T)[2]  # Z-depth in camera 2

    # Count valid 3D points where depth is positive
    num_positive_depth = np.sum((depth1 > 0) & (depth2 > 0))

    # Select the best P2 with the most valid points
    if num_positive_depth > max_positive_depth:
        max_positive_depth = num_positive_depth
        best_P2 = P2
        best_pts3D = pts3D

# Print the best projection matrix
print("Selected Best P2:", best_P2)
print("Max positive depth count:", max_positive_depth)

# 9. Scatter plot the correct 3D points

def scatter_3D_points(points3D):
    """
    Plots a 3D scatter plot from a set of 3D points.
    
    Args:
    - points3D (numpy array): An Nx3 array of 3D points (X, Y, Z)
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates
    x = points3D[:, 0]
    y = points3D[:, 1]
    z = points3D[:, 2]

    # Scatter plot
    ax.scatter(x, y, z, c='b', marker='.', s=10)  # Small blue dots

    # Set labels
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    # View angle to mimic the given image
    ax.view_init(elev=90, azim=90)

    plt.show()


# scatter_3D_points(best_pts3D)



# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
R1 = np.identity(3) # Reference camera so no rotation
t1 = np.zeros(3) # Reference camears so no scaling

R2 = best_P2[: ,:3]
t2 = np.linalg.inv(R2) @ best_P2[:, 3]

np.savez("../data/extrinsics.npz", R1=R1, R2=R2, t1=t1, t2=t2)