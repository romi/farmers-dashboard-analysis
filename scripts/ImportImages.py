import numpy
import glob
import os
import argparse
from farmersdashboard.core import Database, Session, Metadata, CameraInfo
import shutil
from datetime import datetime

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                        help="Path to the config file")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to the directory with the images")
    parser.add_argument("-d", "--db", required=True,
                        help="Path to the database directory")
    args = parser.parse_args()

    images = glob.glob(args.input + '/*.jpg')

    camera_info = CameraInfo(args.config)

    database = Database(args.db)
    session = database.create_session()
    metadata = session.create_metadata()
    metadata.set_camera_info(camera_info)
    metadata.set_origin(None, 'import', f'Import from {args.input}')
        
    j = 0
    for image in images:
        shutil.copy2(image, metadata.make_image_path(j))
        metadata.add_image(metadata.make_image_filename(j))
        j = j + 1

    metadata.store()

    print(metadata.list_image_files())
    print(metadata.get_image('00024.jpg'))
