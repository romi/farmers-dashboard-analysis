# System information:
# - Linux Mint 18.1 Cinnamon 64-bit
# - Python 2.7 with OpenCV 3.2.0

import numpy
import cv2 as cv
from cv2 import aruco
import glob
import argparse
from farmersdashboard.core import CameraInfo

# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 5 
CHARUCOBOARD_SQUARELENGTH = 0.050
CHARUCOBOARD_MARKERLENGTH = 0.030
#ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)
ARUCO_DICT = aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

# Create constants to be passed into OpenCV and Aruco methods
CHARUCO_BOARD = aruco.CharucoBoard((CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT), CHARUCOBOARD_SQUARELENGTH, CHARUCOBOARD_MARKERLENGTH, ARUCO_DICT)


def do_calibration(image_dir):
    # Create the arrays and variables we'll use to store info like corners and IDs from images processed
    corners_all = [] # Corners discovered in all images processed
    ids_all = [] # Aruco ids corresponding to corners discovered
    image_size = None # Determined at runtime


    # This requires a set of images or a video taken with the camera you want to calibrate
    # I'm using a set of images taken with the camera with the naming convention:
    # 'camera-pic-of-charucoboard-<NUMBER>.jpg'
    # All images used should be the same size, which if taken with the same camera
    # shouldn't be a problem

    images = glob.glob(image_dir + '/*.jpg')

    print(images)

    # Make sure at least one image was found
    if len(images) < 1:
        # Calibration failed because there were no images, warn the user
        print("Calibration was unsuccessful. No images were found.")
        # Exit for failure
        raise ValueError("Directory contains no .jpg images")

    count = 0

    # Loop through images glob'ed
    for iname in images:
        # Open the image
        img = cv.imread(iname)
        # Grayscale the image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find aruco markers in the query image
        corners, ids, _ = aruco.detectMarkers(image=gray, dictionary=ARUCO_DICT)

        # Outline the aruco markers found in our query image
        img = aruco.drawDetectedMarkers(image=img, corners=corners)

        # Get charuco corners and ids from detected aruco markers
        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids, image=gray, board=CHARUCO_BOARD)

        # If a Charuco board was found, let's collect image/corner points
        # Requiring at least 20 squares
        if response > 20:
            # Add these corners and ids to our calibration arrays
            corners_all.append(charuco_corners)
            ids_all.append(charuco_ids)
            count = count + 1
        
            # Draw the Charuco board we've detected to show our calibrator
            # the board was properly detected
            img = aruco.drawDetectedCornersCharuco(image=img,
                                                   charucoCorners=charuco_corners,
                                                   charucoIds=charuco_ids)
       
            # If our image size is unknown, set it now
            if not image_size:
                image_size = gray.shape[::-1]
    
            # Reproportion the image, maxing width or height at 1000
            proportion = max(img.shape) / 1000.0
            img = cv.resize(img, (int(img.shape[1]/proportion),
                                  int(img.shape[0]/proportion)))
            # Pause to display each image, waiting for key press
            print(f'Count {count}')
            cv.imshow('Charuco board', img)
            cv.waitKey(0)
        else:
            print("Not able to detect a charuco board in image: {}".format(iname))

    # Destroy any open CV windows
    cv.destroyAllWindows()

    # Make sure we were able to calibrate on at least one charucoboard by checking
    # if we ever determined the image size
    if not image_size:
        # Calibration failed because we didn't see any charucoboards of the PatternSize used
        print("Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")

        # Exit for failure
        raise ValueError("No charucoboards were detected.")

    # Now that we've seen all of our images, perform the camera calibration
    # based on the set of points we've discovered
    calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)

    # cameraMatrix	A 3x3 floating-point camera matrix
    #   A=[[fx, 0, 0],[0, fy, 0], [cx, cy, 1]]. 
    
    # distCoeffs	A vector of distortion coefficients
    #  (k1,k2,p1,p2[,k3[,k4,k5,k6],[s1,s2,s3,s4]]) of 4, 5, 8 or 12 elements
    
    # Print matrix and distortion coefficient to the console
    print(cameraMatrix)
    print(distCoeffs)
    print('Calibration successful.')

    return cameraMatrix, distCoeffs
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                        help="Path to the config file")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to the directory with the chAruco images")
    args = parser.parse_args()

    camera_info = CameraInfo(args.config)
    
    cameraMatrix, distCoeffs = do_calibration(args.input)

    camera_info.set_intrinsics_opencv(cameraMatrix[0][0],
                                      cameraMatrix[1][1],
                                      cameraMatrix[0][2],
                                      cameraMatrix[1][2])
    camera_info.set_distortion_parameters(distCoeffs[0][0], distCoeffs[0][1])


