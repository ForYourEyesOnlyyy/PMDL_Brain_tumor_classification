import os

def get_project_root():
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_file_path, '../../'))

    return root_dir
