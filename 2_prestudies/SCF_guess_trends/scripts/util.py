import os

def get_xyz_file_paths(folder):
    xyz_file_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.xyz'):
                xyz_file_paths.append(os.path.join(root, file))
    return xyz_file_paths