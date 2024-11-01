import ipfshttpclient
from configs import IPFS_URL


class File_Access:
    _ipfs_connect = None

    def __init__(self):
        self._ipfs_connect = ipfshttpclient.connect(IPFS_URL)

    def download(self, cid, output_path):
        self._ipfs_connect.get(cid, target=output_path)

    def upload(self, data_file):
        ret = self._ipfs_connect
        return ret["Hash"]


def get_ipfs_instance():
    if not hasattr(get_ipfs_instance, "_instance"):
        get_ipfs_instance._instance = File_Access()
    return get_ipfs_instance._instance
