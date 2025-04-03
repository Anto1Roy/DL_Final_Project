import os

class StreamingFileManager:
    def __init__(self, remote_base_url, cache_dir="/tmp/ipd_cache"):
        self.remote_base_url = remote_base_url
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.is_local = os.path.exists(remote_base_url)

    def get(self, path_in_remote):
        # if self.is_local:
        full_path = os.path.join(self.remote_base_url, path_in_remote).replace("\\", "/")
        return full_path

        # Never download
        # # Otherwise: download via gdown
        # hashed = hashlib.md5(path_in_remote.encode()).hexdigest()
        # local_path = os.path.join(self.cache_dir, hashed + "_" + os.path.basename(path_in_remote))

        # if not os.path.exists(local_path):
        #     url = f"{self.remote_base_url}/{path_in_remote}"
        #     print(f"[Streaming] Downloading: {url}")
        #     gdown.download(url, local_path, quiet=False)

        # return local_path

    # def remove(self, path):
    #     if not self.is_local and os.path.exists(path):
    #         try:
    #             os.remove(path)
    #         except Exception as e:
    #             print(f"[Warning] Couldn't remove {path}: {e}")
