import os
import zipfile
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import tempfile
from pathlib import Path

class ZipEncryption:
    def __init__(self, password: str, salt: bytes = None):
        """Initialize encryption with a password."""
        self._original_password = password
        self.salt = salt if salt is not None else os.urandom(16)
        self.key = self._generate_key(password.encode(), self.salt)
        self.cipher = Fernet(self.key)
    
    def _generate_key(self, password: bytes, salt: bytes) -> bytes:
        """Generate a secure key from password and salt using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key

    def encrypt_files(self, file_paths: list, output_path: str):
        """Create an encrypted zip file containing the specified files.
        
        Args:
            file_paths (list): List of paths to files to encrypt
            output_path (str): Path where to save the encrypted zip
        """
        # Create a temporary zip file
        with tempfile.NamedTemporaryFile(delete=False) as temp_zip:
            # Create zip archive
            with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in file_paths:
                    # Add file to zip with its basename as name in zip
                    zipf.write(file_path, Path(file_path).name)

        # Read the temporary zip file
        with open(temp_zip.name, 'rb') as f:
            zip_data = f.read()

        # Encrypt the zip data
        encrypted_data = self.cipher.encrypt(zip_data)

        # Save encrypted data and salt
        with open(output_path, 'wb') as f:
            # Write salt first (16 bytes)
            f.write(self.salt)
            # Write encrypted data
            f.write(encrypted_data)

        # Clean up temporary file
        os.unlink(temp_zip.name)

    def decrypt_file(self, input_path: str, output_dir: str):
        """Decrypt an encrypted zip file and extract its contents.
        
        Args:
            input_path (str): Path to encrypted zip file
            output_dir (str): Directory where to extract decrypted files
        """
        # Read encrypted file
        with open(input_path, 'rb') as f:
            # Read salt (first 16 bytes)
            salt = f.read(16)
            # Read the rest as encrypted data
            encrypted_data = f.read()

        # Reinitialize with correct salt
        self.__init__(password=self._original_password, salt=salt)

        # Decrypt the data
        decrypted_data = self.cipher.decrypt(encrypted_data)

        # Create a temporary zip file with decrypted data
        with tempfile.NamedTemporaryFile(delete=False) as temp_zip:
            temp_zip.write(decrypted_data)

        # Extract the zip file
        with zipfile.ZipFile(temp_zip.name, 'r') as zipf:
            zipf.extractall(output_dir)

        # Clean up temporary file
        os.unlink(temp_zip.name) 