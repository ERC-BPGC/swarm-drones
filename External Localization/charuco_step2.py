import cv2
import numpy as np
import os
import glob

# Define board parameters
squares_x = 7  # number of chessboard squares in X direction
squares_y = 5  # number of chessboard squares in Y direction
square_length = 0.059  # in meters or your chosen unit (e.g., 5.9 cm -> 0.059 m)
marker_length = 0.047  # marker length (must be smaller than square_length, e.g., 4.7 cm -> 0.047 m)

# Use predefined dictionary for 4x4 markers (50 unique markers)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# Updated board creation for OpenCV 4.12
board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)

# Calibration criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Storage for calibration
all_corners = []
all_ids = []
image_size = None

# Folder setup
cam_images_folder_name = 'cam_1'
cam_images_folder_name_calibrated = f'{cam_images_folder_name}_c'
os.makedirs(cam_images_folder_name_calibrated, exist_ok=True)

images = glob.glob(f'./{cam_images_folder_name}/*.jpg')
total_images = len(images)
print(f"Found {total_images} images in {cam_images_folder_name} folder")

# Create detector objects
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict)
charuco_detector = cv2.aruco.CharucoDetector(board)

# Process each image
for i, fname in enumerate(images):
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to read image: {fname}")
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]  # Store for calibration
    
    # Create a clean copy for visualization
    display_img = img.copy()
    
    # Detect ArUco markers in the image
    corners, ids, rejected = aruco_detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(display_img, corners, ids)
        
        # Interpolate ChArUco corners
        # Fix for OpenCV 4.12: handle the return value properly
        result = charuco_detector.detectBoard(gray)
        
        # In OpenCV 4.12, detectBoard returns a tuple with (charuco_corners, charuco_ids, ...)
        # Extract corners and IDs from the result
        if isinstance(result, tuple) and len(result) >= 2:
            charuco_corners, charuco_ids = result[0], result[1]
        else:
            # If result is not as expected, try to access it as a single object
            charuco_corners = result
            charuco_ids = None
            
        if charuco_corners is not None and len(charuco_corners) > 0:
            retval = len(charuco_corners)
            
            if retval > 10:  # Require at least 10 valid ChArUco corners
                # Draw the detected ChArUco corners
                cv2.aruco.drawDetectedCornersCharuco(display_img, charuco_corners, charuco_ids)
                
                # Show approval information
                cv2.putText(display_img, f"Image {i+1}/{total_images}: Detected {retval} corners", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_img, "Press 'y' to approve, 'n' to reject, 'q' to quit", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display for approval
                cv2.imshow("ChArUco Detection - Approve?", display_img)
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('y'):  # Approved
                    print(f"Image {i+1}/{total_images}: Approved with {retval} corners")
                    all_corners.append(charuco_corners)
                    all_ids.append(charuco_ids)
                    
                    # Save the annotated image
                    new_frame_name = os.path.join(cam_images_folder_name_calibrated, 
                                                 os.path.basename(fname))
                    cv2.imwrite(new_frame_name, display_img)
                    
                elif key == ord('n'):  # Rejected
                    print(f"Image {i+1}/{total_images}: Rejected")
                elif key == ord('q'):  # Quit
                    print("Calibration process interrupted by user")
                    break
            else:
                # Not enough corners detected
                cv2.putText(display_img, f"Image {i+1}/{total_images}: Not enough corners ({retval})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_img, "Press any key to continue, 'q' to quit", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow("ChArUco Detection - Not enough corners", display_img)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    print("Calibration process interrupted by user")
                    break
        else:
            # Failed to interpolate corners
            cv2.putText(display_img, f"Image {i+1}/{total_images}: Failed to interpolate ChArUco corners", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_img, "Press any key to continue, 'q' to quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("ChArUco Detection - Failed interpolation", display_img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("Calibration process interrupted by user")
                break
    else:
        # No markers detected
        cv2.putText(display_img, f"Image {i+1}/{total_images}: No markers detected", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_img, "Press any key to continue, 'q' to quit", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("ChArUco Detection - No markers", display_img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Calibration process interrupted by user")
            break

cv2.destroyAllWindows()

# Camera calibration using ChArUco
if len(all_corners) > 0:
    print(f"\nPerforming calibration with {len(all_corners)} approved images...")
    
    # Initialize camera matrix and distortion coefficients
    camera_matrix = np.eye(3)
    dist_coeffs = np.zeros((5, 1))
    
    # Setup calibration flags
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + 
             cv2.CALIB_ZERO_TANGENT_DIST)
    
    # Prepare object points for approved charuco corners
    obj_points = []
    img_points = []
    
    # For each approved image, get the object points corresponding to the detected charuco corners
    for i in range(len(all_corners)):
        obj_p = []
        for j in range(len(all_ids[i])):
            idx = int(all_ids[i][j])  # Extract the integer value from the array
            obj_p.append(board.getChessboardCorners()[idx])
        obj_points.append(np.array(obj_p, dtype=np.float32))
        img_points.append(all_corners[i])
    
    # Perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, camera_matrix, dist_coeffs, flags=flags
    )

    # Save calibration results to a file
    calibration_file = f"{cam_images_folder_name}_calibration.npz"
    np.savez(calibration_file, 
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rvecs=rvecs,
             tvecs=tvecs)
    
    print(f"Calibration successful. Results saved to {calibration_file}")
    print("\nCamera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(obj_points)):
        projected_points, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        
        error = cv2.norm(img_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
        mean_error += error
    
    print(f"\nMean Reprojection Error: {mean_error/len(obj_points)} pixels")
else:
    print("Not enough approved images for calibration.")
