import cv2
import numpy as np
import os
import glob
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_camera_calibration(calibration_file):
    """Load camera calibration data from a .npz file"""
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
    
    data = np.load(calibration_file)
    return data['camera_matrix'], data['dist_coeffs']

def create_charuco_board():
    """Create the ChArUco board with the same parameters as in calibration"""
    # Define board parameters - match with charuco_step2.py
    squares_x = 7
    squares_y = 5
    square_length = 0.059  # in meters
    marker_length = 0.047  # in meters
    
    # Use predefined dictionary for 4x4 markers (50 unique markers)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Create the CharUco board
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    
    return board, aruco_dict

def estimate_board_pose(image_path, camera_matrix, dist_coeffs, board, aruco_dict, debug=False):
    """Estimate the pose of the ChArUco board relative to the camera"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create detector objects
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict)
    charuco_detector = cv2.aruco.CharucoDetector(board)
    
    # Detect ArUco markers
    corners, ids, rejected = aruco_detector.detectMarkers(gray)
    
    if ids is None or len(ids) == 0:
        if debug:
            print(f"No ArUco markers detected in image: {image_path}")
            if os.path.exists(image_path):
                debug_img = img.copy()
                cv2.putText(debug_img, "NO MARKERS DETECTED", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                debug_dir = os.path.join(os.path.dirname(os.path.dirname(image_path)), "debug")
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"no_markers_{os.path.basename(image_path)}"), debug_img)
        raise ValueError(f"No ArUco markers detected in image: {image_path}")
    
    # Create a debug image if requested
    if debug:
        debug_img = img.copy()
        cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)
    
    # Detect ChArUco corners
    result = charuco_detector.detectBoard(gray)
    
    # Extract ChArUco corners and IDs
    if isinstance(result, tuple) and len(result) >= 2:
        charuco_corners, charuco_ids = result[0], result[1]
    else:
        charuco_corners = result
        charuco_ids = None
    
    if charuco_corners is None or len(charuco_corners) == 0:
        if debug:
            print(f"No ChArUco corners detected in image: {image_path}")
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(image_path)), "debug")
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"no_corners_{os.path.basename(image_path)}"), debug_img)
        raise ValueError(f"No ChArUco corners detected in image: {image_path}")
    
    if debug:
        cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_ids)
        debug_dir = os.path.join(os.path.dirname(os.path.dirname(image_path)), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"detected_{os.path.basename(image_path)}"), debug_img)
    
    # Prepare object points for detected charuco corners
    obj_points = []
    for idx in charuco_ids:
        if isinstance(idx, np.ndarray):
            idx = idx.item()  # Convert from numpy array to scalar
        obj_points.append(board.getChessboardCorners()[idx])
    obj_points = np.array(obj_points, dtype=np.float32)
    
    # Estimate pose of the board with respect to the camera
    retval, rvec, tvec = cv2.solvePnP(obj_points, charuco_corners, camera_matrix, dist_coeffs)
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    return R, tvec, img.shape[:2][::-1]  # Return image width, height for visualization

def transform_to_world_coordinates(R_board_to_cam, t_board_to_cam):
    """
    Transform from board-to-camera to camera-to-world coordinates.
    The ChArUco board defines the world coordinate system.
    """
    # The camera-to-world transformation is the inverse of board-to-camera
    R_cam_to_world = R_board_to_cam.T
    t_cam_to_world = -R_cam_to_world.dot(t_board_to_cam)
    
    return R_cam_to_world, t_cam_to_world

def compute_world_to_camera_matrix(R_cam_to_world, t_cam_to_world):
    """Compute the world-to-camera transformation matrix"""
    T = np.eye(4)
    T[:3, :3] = R_cam_to_world.T  # Transpose for world-to-camera rotation
    T[:3, 3] = -R_cam_to_world.T.dot(t_cam_to_world).flatten()
    
    return T

def visualize_camera_poses(camera_poses, image_sizes=None):
    """Visualize the camera poses in 3D"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Camera visualization parameters
    camera_scale = 0.2
    
    # World coordinate frame
    origin = np.zeros(3)
    x_axis = np.array([1, 0, 0]) * camera_scale
    y_axis = np.array([0, 1, 0]) * camera_scale
    z_axis = np.array([0, 0, 1]) * camera_scale
    
    ax.quiver(*origin, *x_axis, color='r', label='X-axis')
    ax.quiver(*origin, *y_axis, color='g', label='Y-axis')
    ax.quiver(*origin, *z_axis, color='b', label='Z-axis')
    
    # Plot each camera
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, pose in enumerate(camera_poses):
        R = np.array(pose['R'])
        t = np.array(pose['t'])
        
        # Camera position in world coordinates
        cam_pos = t
        
        # Camera orientation vectors
        cam_x = R[:, 0] * camera_scale
        cam_y = R[:, 1] * camera_scale
        cam_z = R[:, 2] * camera_scale
        
        color = colors[i % len(colors)]
        
        # Plot camera position
        ax.scatter(*cam_pos, color=color, s=100, label=f'Camera {i+1}')
        
        # Plot camera orientation
        ax.quiver(*cam_pos, *cam_x, color='r', alpha=0.5)
        ax.quiver(*cam_pos, *cam_y, color='g', alpha=0.5)
        ax.quiver(*cam_pos, *cam_z, color='b', alpha=0.5)
        
        # If image sizes are provided, visualize the image plane
        if image_sizes and i < len(image_sizes):
            w, h = image_sizes[i]
            # Normalize to camera scale
            aspect_ratio = h / w
            width = camera_scale
            height = width * aspect_ratio
            
            # Image plane corners in camera coordinates
            corners = np.array([
                [-width/2, -height/2, camera_scale],
                [width/2, -height/2, camera_scale],
                [width/2, height/2, camera_scale],
                [-width/2, height/2, camera_scale]
            ])
            
            # Transform corners to world coordinates
            world_corners = []
            for corner in corners:
                world_corner = R.dot(corner) + t
                world_corners.append(world_corner)
            
            # Create a polygon for the image plane
            poly = np.array(world_corners)
            ax.plot([poly[0, 0], poly[1, 0]], [poly[0, 1], poly[1, 1]], [poly[0, 2], poly[1, 2]], color=color)
            ax.plot([poly[1, 0], poly[2, 0]], [poly[1, 1], poly[2, 1]], [poly[1, 2], poly[2, 2]], color=color)
            ax.plot([poly[2, 0], poly[3, 0]], [poly[2, 1], poly[3, 1]], [poly[2, 2], poly[3, 2]], color=color)
            ax.plot([poly[3, 0], poly[0, 0]], [poly[3, 1], poly[0, 1]], [poly[3, 2], poly[0, 2]], color=color)
    
    # Configure plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Poses in World Coordinates')
    ax.legend()
    
    # Set equal scaling for all axes
    max_range = np.array([
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        ax.get_zlim()[1] - ax.get_zlim()[0]
    ]).max() / 2.0
    
    mid_x = (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2
    mid_y = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
    mid_z = (ax.get_zlim()[1] + ax.get_zlim()[0]) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig('camera_poses_3d.png', dpi=300)
    plt.show()

def find_best_images(base_dir, debug=True):
    """Find the best images that have visible ArUco markers in each camera folder"""
    camera_dirs = sorted(os.listdir(base_dir))
    camera_dirs = [d for d in camera_dirs if d.startswith('cam_')]
    
    best_images = {}
    
    for cam_dir in camera_dirs:
        cam_path = os.path.join(base_dir, cam_dir)
        if not os.path.isdir(cam_path):
            continue
            
        # Only get jpg/png files
        images = glob.glob(os.path.join(cam_path, '*.jpg')) + glob.glob(os.path.join(cam_path, '*.png'))
        
        if not images:
            print(f"No images found in {cam_path}")
            continue
            
        # Sort images by name
        images.sort()
        
        # Try to find an image with ArUco markers for this camera
        if debug:
            print(f"Searching for ArUco markers in {len(images)} images from {cam_dir}...")
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict)
        
        for img_path in images:
            try:
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = aruco_detector.detectMarkers(gray)
                
                if ids is not None and len(ids) >= 4:  # Require at least 4 markers for a good pose estimation
                    if debug:
                        print(f"Found {len(ids)} markers in {os.path.basename(img_path)}")
                    best_images[cam_dir] = img_path
                    break
            except Exception as e:
                if debug:
                    print(f"Error processing {img_path}: {e}")
    
    return best_images

def main():
    # Base directory for synchronized images
    base_dir = os.path.expanduser("~/Pictures/dart_test (jhinu)/Low-Cost-Mocap/computer_code/api/calibrate/syncro_pics")
    calibration_dir = os.path.dirname(base_dir)
    
    # Create debug directory
    debug_dir = os.path.join(calibration_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Enable debug mode
    debug = True
    
    # Find the best images for each camera
    print("Searching for images with ArUco markers...")
    best_images = find_best_images(base_dir, debug=debug)
    
    if not best_images:
        print("No suitable images found with ArUco markers!")
        return
    
    # Print the selected images
    print("\nSelected images for pose estimation:")
    for cam, img_path in best_images.items():
        print(f"{cam}: {os.path.basename(img_path)}")
    
    # Create CharUco board
    board, aruco_dict = create_charuco_board()
    
    # Prepare to store camera poses and image sizes
    camera_poses = []
    image_sizes = []
    world_to_camera_matrices = {}
    
    # Set the first camera as the world origin
    first_cam = sorted(best_images.keys())[0]
    first_camera_pose = {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 0]}
    camera_poses.append(first_camera_pose)
    
    # For visualization, add the image size of the first camera
    first_img = cv2.imread(best_images[first_cam])
    if first_img is not None:
        image_sizes.append(first_img.shape[:2][::-1])  # (width, height)
    else:
        image_sizes.append((1920, 1080))  # Default size if image can't be loaded
    
    # Process each camera (except the first one which is the origin)
    camera_dirs = sorted(best_images.keys())
    for i, cam_dir in enumerate(camera_dirs):
        if i == 0:  # Skip the first camera as it's the origin
            continue
            
        # Load camera calibration
        calibration_file = os.path.join(calibration_dir, f"{cam_dir}_calibration.npz")
        
        try:
            camera_matrix, dist_coeffs = load_camera_calibration(calibration_file)
        except FileNotFoundError:
            print(f"Calibration file not found for camera {cam_dir}. Skipping.")
            continue
        
        img_path = best_images[cam_dir]
        
        try:
            # Get board pose relative to camera
            R_board_to_cam, t_board_to_cam, img_size = estimate_board_pose(
                img_path, camera_matrix, dist_coeffs, board, aruco_dict, debug=debug
            )
            
            # Convert to camera pose in world coordinates (board coordinates)
            R_cam_to_world, t_cam_to_world = transform_to_world_coordinates(
                R_board_to_cam, t_board_to_cam
            )
            
            # Store the camera pose and image size
            camera_poses.append({
                "R": R_cam_to_world.tolist(),
                "t": t_cam_to_world.flatten().tolist()
            })
            image_sizes.append(img_size)
            
            # Compute world-to-camera transformation matrix
            world_to_camera = compute_world_to_camera_matrix(R_cam_to_world, t_cam_to_world)
            world_to_camera_matrices[cam_dir] = world_to_camera
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Output the camera poses
    if len(camera_poses) > 1:
        print("\nCamera Poses:")
        print(json.dumps(camera_poses, indent=2))
        
        # Save the camera poses to a file
        poses_file = os.path.join(calibration_dir, "camera_poses.json")
        with open(poses_file, "w") as f:
            json.dump(camera_poses, f, indent=2)
        print(f"Camera poses saved to {poses_file}")
        
        # Save each world-to-camera matrix
        for cam_dir, matrix in world_to_camera_matrices.items():
            matrix_file = os.path.join(calibration_dir, f"{cam_dir}_world_to_camera_matrix.npy")
            np.save(matrix_file, matrix)
            
            # Also save in a readable format
            txt_file = os.path.join(calibration_dir, f"{cam_dir}_world_to_camera_matrix.txt")
            with open(txt_file, "w") as f:
                f.write(np.array2string(matrix, precision=16))
            print(f"World-to-camera matrix for {cam_dir} saved to {matrix_file} and {txt_file}")
        
        # Visualize the camera poses
        print("\nVisualizing camera poses...")
        visualize_camera_poses(camera_poses, image_sizes)
    else:
        print("\nWarning: Only one camera pose was computed. Cannot establish relative positions.")

if __name__ == "__main__":
    main()
