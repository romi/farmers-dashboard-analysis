import numpy
import glob
import os
import argparse
from farmersdashboard.core import Database, Session, Metadata, CameraInfo
import shutil
from datetime import datetime
import cv2

        
def undistort_image_file(input_file, output_file, camera):
    image = cv2.imread(input_file)
    image = undistort_image(image, camera)
    cv2.imwrite(output_file, image)


def undistort_image(image, camera):
    intrinsics = camera.get_intrinsics_opencv()
    distort_parameters = camera.get_distortion_parameters()
    return cv2.undistort(image, intrinsics, distort_parameters)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--id", required=True,
                        help="The ID of the directory with the images")
    parser.add_argument("-d", "--db", required=True,
                        help="Path to the database directory")
    args = parser.parse_args()

    
    db = Database(args.db)
    input_session = db.get_session(args.id)
    input_metadata = input_session.get_metadata()

    output_session = db.create_session()
    output_metadata = output_session.create_metadata()
    
    camera_info = input_metadata.get_camera_info()
    output_metadata.set_camera_info(camera_info)
    
    filenames = input_metadata.list_image_files()
    
    for filename in filenames:
        input_image = input_metadata.get_image_path(filename)
        output_image = output_metadata.get_image_path(filename)
        undistort_image_file(input_image, output_image, camera_info)
        #shutil.copy2(input_image, output_image)
        output_metadata.add_image(filename)

    output_metadata.set_origin(args.id, 'undistort')
    output_metadata.store()

