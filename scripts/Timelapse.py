import time
import cv2
import argparse
from romi.camera import Camera
from farmersdashboard.core import Database, Session, Metadata, CameraInfo
from PIL import Image
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--topic', type=str, nargs='?', default='camera',
                        help='The registry topic of the camera')
    parser.add_argument('-i', '--interval', type=float, nargs='?', default=1.0,
                        help='The interval between images')
    parser.add_argument("-d", "--db", required=True,
                        help="Path to the databse directory")
    args = parser.parse_args()

    
    #
    print('Connecting to camera')
    camera = Camera(args.topic)
    camera.power_up()

    db = Database(args.db)
    session = db.create_session()
    metadata = session.create_metadata()

    count = 0
    print('Starting timelapse')
    while True:
        filename = metadata.make_image_filename(count)
        path = metadata.make_image_path(count)
        print(f'Grabbing camera image')
        image = camera.grab()
        print(f'Storing to {path}')
        #cv2.imwrite(path, image)
        image.save(path)
        metadata.add_image(filename)
        count = count + 1
        print(f'Stored {filename}')
        metadata.store()
        time.sleep(args.interval)
