import os

def make_folder_if_not_exist(base_path, folder_name):
    folder_path= os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path