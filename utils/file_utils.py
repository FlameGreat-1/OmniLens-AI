import os
import json
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

def create_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def list_files(directory, extensions=None):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if extensions is None or any(filename.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
    return files

def save_image(image, file_path, quality=95):
    extension = os.path.splitext(file_path)[1].lower()
    if extension == '.jpg' or extension == '.jpeg':
        cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif extension == '.png':
        cv2.imwrite(file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    elif extension == '.tiff':
        cv2.imwrite(file_path, image)
    else:
        raise ValueError(f"Unsupported file format: {extension}")

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def create_backup(file_path, backup_dir):
    create_directory(backup_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"{os.path.basename(file_path)}_{timestamp}")
    shutil.copy2(file_path, backup_path)
    return backup_path

def compress_files(file_list, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in file_list:
            zipf.write(file, os.path.basename(file))

def decompress_files(zip_path, output_dir):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(output_dir)

def get_file_info(file_path):
    stat = os.stat(file_path)
    return {
        "name": os.path.basename(file_path),
        "size": stat.st_size,
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "extension": os.path.splitext(file_path)[1]
    }

def rename_file(old_path, new_path):
    os.rename(old_path, new_path)

def delete_file(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

def get_unique_filename(directory, base_name, extension):
    counter = 1
    new_name = f"{base_name}{extension}"
    while os.path.exists(os.path.join(directory, new_name)):
        new_name = f"{base_name}_{counter}{extension}"
        counter += 1
    return new_name

def create_symlink(source, link_name):
    os.symlink(source, link_name)

def copy_file(src, dst):
    shutil.copy2(src, dst)

def move_file(src, dst):
    shutil.move(src, dst)
