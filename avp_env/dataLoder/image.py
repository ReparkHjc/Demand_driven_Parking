import os
import cv2
from avp_env.dataLoder.path import PathLoader
import numpy as np

class ImageLoader:
    def __init__(self, env_type, image_shape):
        self.path_loader = PathLoader(env_type)
        self.experiment_paths = self.path_loader.load_path()
        self.image_shape = image_shape
        self.image_data = self._load_images()
        self.render_image = self._load_render_images()
        self.image_side_data = self._load_side_images()
        self.image_right_data = self._load_right_images()

    def _load_images(self):
        image_data = {}
        for experiment_path in self.experiment_paths:
            # get park_id 和 experiment_id
            park_num = os.path.basename(os.path.dirname(experiment_path))  # Park_1
            park_id = park_num.split('_')[-1]  #  '1'

            experiment_id = os.path.basename(experiment_path)             # 20240423

            cam_folders = [os.path.join(experiment_path, f"cam{i}") for i in range(4)]

            file_names = sorted(os.listdir(cam_folders[0]))

            for filename in file_names:
                if filename.endswith('.jpg') or filename.endswith('.JPG'):
                    images = []
                    for cam_folder in cam_folders:
                        file_path = os.path.join(cam_folder, filename)
                        image_array = cv2.imread(file_path)
                        resized_image = cv2.resize(image_array, (self.image_shape[1], self.image_shape[0]),
                                               interpolation=cv2.INTER_AREA)
                        images.append(resized_image)

                    # images = np.stack(images, axis=0)    # (4, H, W, 3)
                    images = np.concatenate(images, axis=-1)  # (H, W, 12)

                    # Use a unique key combining experiment_id and filename
                    unique_key = f"{park_id}/{experiment_id}/{filename}"
                    image_data[unique_key] = images
        return image_data

    def _load_side_images(self):
        image_side_data = {}
        for experiment_path in self.experiment_paths:
            # get park_id 和 experiment_id
            park_num = os.path.basename(os.path.dirname(experiment_path))  # Park_1
            park_id = park_num.split('_')[-1]  #  '1'

            experiment_id = os.path.basename(experiment_path)             # 20240423

            cam_folders = [os.path.join(experiment_path, f"cam{i}") for i in [1,2]] # the side img

            file_names = sorted(os.listdir(cam_folders[0]))

            for filename in file_names:
                if filename.endswith('.jpg') or filename.endswith('.JPG'):
                    images = []
                    for cam_folder in cam_folders:
                        file_path = os.path.join(cam_folder, filename)
                        image_array = cv2.imread(file_path)
                        resized_image = cv2.resize(image_array, (self.image_shape[1], self.image_shape[0]),
                                               interpolation=cv2.INTER_AREA)
                        images.append(resized_image)

                    # images = np.stack(images, axis=0)    # (4, H, W, 3)
                    images = np.concatenate(images, axis=-1)  # (H, W, 6)

                    # Use a unique key combining experiment_id and filename
                    unique_key = f"{park_id}/{experiment_id}/{filename}"
                    image_side_data[unique_key] = images
        return image_side_data

    def _load_right_images(self):
        image_side_data = {}
        for experiment_path in self.experiment_paths:
            # get park_id 和 experiment_id
            park_num = os.path.basename(os.path.dirname(experiment_path))  # Park_1
            park_id = park_num.split('_')[-1]  #  '1'

            experiment_id = os.path.basename(experiment_path)             # 20240423

            cam_folders = [os.path.join(experiment_path, f"cam{i}") for i in [1]] # the side img

            file_names = sorted(os.listdir(cam_folders[0]))

            for filename in file_names:
                if filename.endswith('.jpg') or filename.endswith('.JPG'):
                    images = []
                    for cam_folder in cam_folders:
                        file_path = os.path.join(cam_folder, filename)
                        image_array = cv2.imread(file_path)
                        resized_image = cv2.resize(image_array, (self.image_shape[1], self.image_shape[0]),
                                               interpolation=cv2.INTER_AREA)
                        images.append(resized_image)

                    # images = np.stack(images, axis=0)    # (4, H, W, 3)
                    images = np.concatenate(images, axis=-1)  # (H, W, 3)

                    # Use a unique key combining experiment_id and filename
                    unique_key = f"{park_id}/{experiment_id}/{filename}"
                    image_side_data[unique_key] = images
        return image_side_data

    def _load_render_images(self):
        render_image = {}
        for experiment_path in self.experiment_paths:
            park_num = os.path.basename(os.path.dirname(experiment_path))  # Park_1
            park_id = park_num.split('_')[-1]  # '1'
            experiment_id = os.path.basename(experiment_path)  # 20240423

            # cam_folder = os.path.join(experiment_path, "cam0")
            cam_folders = [os.path.join(experiment_path, f"cam{i}") for i in range(4)]

            file_names = sorted(os.listdir(cam_folders[0]))

            # for filename in file_names:
            #     if filename.endswith('.jpg') or filename.endswith('.JPG'):
            #         filepath = os.path.join(cam_folder, filename)
            #         img = cv2.imread(filepath)
            #         render_image[f"{park_id}/{experiment_id}/{filename}"] = img
            for filename in file_names:
                if filename.endswith('.jpg') or filename.endswith('.JPG'):
                    images = []
                    for cam_folder in cam_folders:
                        file_path = os.path.join(cam_folder, filename)
                        image_array = cv2.imread(file_path)
                        resized_image = cv2.resize(image_array, (self.image_shape[1], self.image_shape[0]),
                                               interpolation=cv2.INTER_AREA)
                        images.append(resized_image)

                    images = np.concatenate(images, axis=-1)  # (H, W, 12)

                    # Use a unique key combining experiment_id and filename
                    unique_key = f"{park_id}/{experiment_id}/{filename}"
                    render_image[unique_key] = images
        return render_image
