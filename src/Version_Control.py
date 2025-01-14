import hashlib
import os
import json

class VersionControl:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.versions = {}
        self.load_versions()

    def add_version(self, file_path, description):
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        version = {
            'hash': file_hash,
            'description': description,
            'timestamp': datetime.now().isoformat()
        }
        
        if file_path not in self.versions:
            self.versions[file_path] = []
        
        self.versions[file_path].append(version)
        self.save_versions()

        # Copy file to versions directory
        version_path = os.path.join(self.base_dir, file_hash)
        shutil.copy2(file_path, version_path)

    def get_versions(self, file_path):
        return self.versions.get(file_path, [])

    def revert_to_version(self, file_path, version_hash):
        versions = self.get_versions(file_path)
        for version in versions:
            if version['hash'] == version_hash:
                version_path = os.path.join(self.base_dir, version_hash)
                shutil.copy2(version_path, file_path)
                return True
        return False

    def save_versions(self):
        with open(os.path.join(self.base_dir, 'versions.json'), 'w') as f:
            json.dump(self.versions, f)

    def load_versions(self):
        version_file = os.path.join(self.base_dir, 'versions.json')
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                self.versions = json.load(f)

