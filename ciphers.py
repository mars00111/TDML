import json
import numpy as np

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend


class Ciphers:
    _public_key = None
    _private_key = None

    def __init__(self):
        self._private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        self._public_key = self._private_key.public_key()

    def public_key_bytes(self):
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def save_key(self, filename):
        with open(filename, "wb") as f:
            if isinstance(self._private_key, rsa.RSAPrivateKey):
                f.write(
                    self._private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.TraditionalOpenSSL,
                    )
                )
            else:
                f.write(
                    self._public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo,
                    )
                )

    def encrypt_data(self, data):
        if isinstance(data, dict) or isinstance(data, list):
            data = json.dumps(data).encode("utf-8")

        encrypted_data = self._private_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return encrypted_data

    def encrypt_dataset(self, dataset):
        data_bytes = dataset.tobytes()
        encrypted_data = self.encrypt_data(data_bytes)
        return encrypted_data

    def decrypt_data(self, encrypted_data):
        decrypted_data = self._public_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        # decrypted_data = json.loads(decrypted_data.decode('utf-8'))
        return decrypted_data

    def decrypt_dataset(self, encrypted_data):
        decrypted_data = self.decrypt_data(decrypted_data)
        return np.frombuffer(decrypted_data, dtype=np.float64).reshape(self.data.shape)


def get_ciphers_instance():
    if not hasattr(get_ciphers_instance, "_instance"):
        get_ciphers_instance._instance = Ciphers()
    return get_ciphers_instance._instance
