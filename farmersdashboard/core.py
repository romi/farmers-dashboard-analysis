import json
import numpy as np
import os
import cv2 as cv
import sys
import math
from datetime import datetime


class CablebotInstallation:
    def __init__(self, length, height_start, height_end, max_bending, camera_length):
        self.__check_values(length, height_start, height_end, max_bending, camera_length)
        self.length = length
        self.height_start = height_start
        self.height_end = height_end
        self.max_bending = max_bending
        self.camera_length = camera_length

        
    def __check_values(self, length, height_start, height_end,
                       max_bending, camera_length):
        assert length > 1.0 and length < 200.0
        assert height_start > 0.5 and height_start < 20.0
        assert height_end > 0.5 and height_end < 20.0
        assert (max_bending >= 0.0
                and max_bending < height_start
                and max_bending < height_end )
        assert (camera_length >= 0.0
                and camera_length <= 1.0
                and camera_length < height_start
                and camera_length < height_end)
        
    def get_z_straight_cable(self, x):
        z0 = self.height_start
        z1 = self.height_end
        alpha = x / self.length
        z_cable = z0 * (1.0 - alpha) + z1 * alpha
        return z_cable

    
    def get_z_bending(self, x):
        h = self.max_bending
        L2 = self.length / 2.0
        w = math.sqrt(L2*L2 + h*h)
        xe = x - L2
        dz = h * math.sqrt(1 - (xe*xe) / (w*w))
        return dz

    
    def get_phi_y_bending(self, x):
        dz_bending = self.get_z_bending(x)
        # Angle around y-axis, due the bending of the cable
        alpha1 = math.atan2(dz_bending, x)
        alpha2 = math.atan2(dz_bending, self.length - x)
        phi_y = (alpha2 - alpha1) / 2.0
        return phi_y

    
    def get_phi_x(self, z_cablebot, dy_soil, phi_y):
        phi_x = math.atan2(dy_soil, z_cablebot / math.cos(phi_y))
        return phi_x

    
    def estimate_homography(self, x, dy_soil):
        phi_y = self.get_phi_y_bending(x)
        z_cable = self.get_z_straight_cable(x)
        dz_bending = self.get_z_bending(x)
        z_cablebot = z_cable - dz_bending
        phi_x = self.get_phi_x(z_cablebot, dy_soil, phi_y)
        return self._estimate_homography(x, phi_x, phi_y, dz_bending)

    
    def _estimate_homography(self, x, phi_x, phi_y, dz_bending):
        # Because of phi_x, the z-offset of the camera in the xz-plane is reduced  
        length_camera_xz = self.camera_length * math.cos(phi_x)

        # 
        alpha1 = math.atan2(dz_bending, x)
        dx_bending = -(x - x * math.cos(alpha1))
        dx_camera = length_camera_xz * math.sin(phi_y) # phi_y <0 => dx < 0
        dx = dx_bending + dx_camera

        # Because of phi_x, the camera is off-center from the cable
        dy_camera = self.camera_length * math.cos(phi_y) * math.sin(phi_x)
        dy = dy_camera
        
        # 
        dz_camera = -(self.camera_length - length_camera_xz * math.cos(phi_y))
        dz = dz_bending + dz_camera
        
        Rx = np.array([[1.0, 0.0, 0.0 ],
                       [0.0, math.cos(phi_x), math.sin(phi_x)],
                       [0.0, -math.sin(phi_x), math.cos(phi_x)]])
        
        Ry = np.array([[math.cos(phi_y), 0.0, math.sin(phi_y)],
                       [0.0,           1.0, 0.0],
                       [-math.sin(phi_y), 0.0, math.cos(phi_y)]])
        R = np.matmul(Ry, Rx)
        
        T = np.array([[dx], [dy], [dz]])
        
        angles = np.array([phi_x, phi_y, 0.0])
        
        return R, T, angles

    
class Database:
    def __init__(self, directory):
        self._directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        
    def create_session(self):
        newid = self._find_id()
        session_dir = os.path.join(self._directory, newid)
        os.makedirs(session_dir)
        return Session(newid, session_dir)
        
    def get_session(self, id):
        session_dir = os.path.join(self._directory, id)
        if not os.path.exists(session_dir):
            raise ValueError(f'Bad session ID {id}, missing directory {session_dir}')
        return Session(id, session_dir)
        
    def _find_id(self):
        i = 0
        while True:
            dirname = f'{i:06d}'
            i = i + 1
            path = os.path.join(self._directory, dirname)
            if not os.path.exists(path):
                return dirname

class Session:
    def __init__(self, id, directory):
        self._id = id
        self._directory = directory

    def create_metadata(self):
        m = Metadata(self._id, datetime.now().isoformat(), self._directory)
        m.store()
        return m

    def get_metadata(self):
        m = Metadata(None, None, self._directory)
        m.load()
        return m

    
class Location:
    def __init__(self, x = 0, y = 0, z = 0, ax = 0, ay = 0, az = 0):
        self.x = x
        self.y = y
        self.z = z
        self.ax = ax
        self.ay = ay
        self.az = az

    def to_json(self):
        return {
            'x': self.x, 'y': self.y, 'z': self.z,
            'ax': self.ax, 'ay': self.ay, 'az': self.az
        }

    def from_json(self, data):
        self.x = data['x']
        self.y = data['y']
        self.z = data['z']
        self.ax = data['ax']
        self.ay = data['ay']
        self.az = data['az']
        
        
class Metadata:
    def __init__(self, id = "", date = "", directory = None):
        self.data = {
            'id': id,
            'date': date
        }
        self._directory = directory
        
    def set_directory(self, directory):
        self._directory = directory
        
    def load(self, directory = None):
        if directory != None:
            self._directory = directory
        if self._directory == None:
            raise ValueError(f'Metadata does not have a directory assigned, yet.')
        path = self._make_path()
        self._load_file(path)

    def _make_path(self):
        return os.path.join(self._directory, 'metadata.json')

    def _load_file(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)

    def store(self):
        with open(self._make_path(), 'w') as f:
            json.dump(self.data, f, indent=4)

    def list_image_files(self):
        return [x['file'] for x in self.data['images']]

    def list_images(self):
        return self.data['images']

    def get_image(self, filename):
        for x in self.list_images():
            if (x['file'] == filename):
                return x
        raise ValueError(f'Unknown image: {filename}')
            
    def get_image_position(self, filename):
        image_data = self.get_image(filename)
        x = image_data['location']['x']
        return x
            
    def get_image_path(self, filename):
        return os.path.join(self._directory, filename)

    def make_image_filename(self, index):
        return f'{index:05d}.jpg'

    def make_image_path(self, index):
        filename = self.make_image_filename(index)
        return os.path.join(self._directory, filename)

    def get_camera_info(self):
        camera_info = CameraInfo()
        camera_info.load_json(self.data['camera'])
        return camera_info

    def set_camera_info(self, camera_info):
        self.data['camera'] = camera_info.to_json()

    def add_image(self, filename, location = None):
        if not 'images' in self.data:
            self.data['images'] = []
        if location:
            self.data['images'].append({'file': filename, 'location': location.to_json()})
        else:
            self.data['images'].append({'file': filename})

    def set_origin(self, id, operation, comment = ""):
        self.data['origin'] = {
            'id': id,
            'operation': operation,
            'comment': comment
        }
        
            
class CameraInfo:
    def __init__(self, path = None):
        self.data = {}
        if path:
            self.load_file(path)

    def load_file(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.load_json(data['camera'])

    def load_json(self, data):
        self.data = data
        type = data['type']
        settings = data[type]
        self.width = settings['width']
        self.height = settings['height']
        
        self.fx_pix = data['intrinsics']['fx']
        self.fy_pix = data['intrinsics']['fy']
        self.cx = data['intrinsics']['cx']
        self.cy = data['intrinsics']['cy']
        self.sensor_dim_x = data['sensor']['dimensions'][0]
        self.sensor_dim_y = data['sensor']['dimensions'][1]
        self.sensor_res_x = data['sensor']['resolution'][0]
        self.sensor_res_y = data['sensor']['resolution'][1]
        self.pixel_size_x = self.sensor_dim_x / self.sensor_res_x
        self.pixel_size_y = self.sensor_dim_y / self.sensor_res_y
        self.fx = self.fx_pix * self.pixel_size_x
        self.fy = self.fy_pix * self.pixel_size_y
        
        distortion = data['distortion']
        distortion_type = distortion['type']
        if distortion_type != 'simple-radial':
            raise ValueError('Only simple-radial distorsion is handled')
        self.k1 = distortion['values'][0]
        self.k2 = distortion['values'][1]

        
    def to_json(self):
        return self.data

        
    def get_pixel_size(self):
        return self.pixel_size_x, self.pixel_size_y


    def get_image_size(self):
        return self.width, self.height


    def get_central_point(self):
        return np.array([cx, cy])


    def get_focal_lengths_meter(self):
        return self.fx, self.fy


    def get_intrinsics_matrix(self):
        return np.array([[self.fx, 0, 0, 0],
                         [0, self.fy, 0, 0],
                         [0, 0, 1, 0]])


    def get_intrinsics_opencv(self):
        return np.matrix([[self.fx_pix, 0, self.cx],
                          [0, self.fy_pix, self.cy],
                          [0, 0, 1]])


    # FIXME
    def set_intrinsics_opencv(self, fx, fy, cx, cy):
        print(f'fx={fx}')
        print(f'fy={fy}')
        print(f'cx={cx}')
        print(f'cy={cy}')
        self.fx_pix = fx
        self.fy_pix = fy
        self.cx = cx
        self.cy = cy
        self.data['intrinsics']['fx'] = fx
        self.data['intrinsics']['fy'] = fy
        self.data['intrinsics']['cx'] = cx
        self.data['intrinsics']['cy'] = cy


    def get_inverse_intrinsics(self):
        return np.matrix([[1.0, 0.0, -self.cx],
                          [0.0, 1.0, -self.cy],
                          [0.0, 0.0, self.fx_pix], # fy?
                          [0, 0, 1]])


    def get_distortion_parameters(self):
        return np.array([self.k1, self.k2, 0.0, 0.0])

    # FIXME
    def set_distortion_parameters(self, k1, k2):
        print(f'k1={k1}')
        print(f'k2={k2}')
        self.k1 = k1
        self.k2 = k2
        self.data['distortion']['values'][0] = k1
        self.data['distortion']['values'][1] = k2


