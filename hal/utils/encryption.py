import os
import zipfile
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import tempfile
from pathlib import Path
import json
from .json_encryption import JsonEncryption

class ZipEncryption:
    def __init__(self, password: str):
        """Initialize encryption with a password."""
        self.json_encryptor = JsonEncryption(password)
    
    def encrypt_files(self, file_paths: list, output_path: str):
        """Create an encrypted zip file containing the specified files.
        
        Args:
            file_paths (list): List of paths to files to encrypt
            output_path (str): Path where to save the encrypted zip
        """
        # Create a temporary directory for encrypted files
        with tempfile.TemporaryDirectory() as temp_dir:
            encrypted_files = []
            
            # Encrypt each JSON file
            for file_path in file_paths:
                file_name = Path(file_path).name
                encrypted_path = Path(temp_dir) / f"{file_name}.encrypted"
                
                # Encrypt the JSON file
                self.json_encryptor.encrypt_json_file(file_path, encrypted_path)
                encrypted_files.append(encrypted_path)
            
            # Create zip archive containing the encrypted files
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for encrypted_file in encrypted_files:
                    zipf.write(encrypted_file, encrypted_file.name)

    def decrypt_file(self, input_path: str, output_dir: str):
        """Decrypt an encrypted zip file and extract its contents.
        
        Args:
            input_path (str): Path to encrypted zip file
            output_dir (str): Directory where to extract decrypted files
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Extract and decrypt files from zip
        with zipfile.ZipFile(input_path, 'r') as zipf:
            for file_name in zipf.namelist():
                # Extract encrypted file
                with zipf.open(file_name) as f:
                    encrypted_data = json.load(f)
                
                # Decrypt the file
                output_path = Path(output_dir) / file_name.replace('.encrypted', '.json')
                self.json_encryptor.decrypt_json_file(
                    encrypted_data['encrypted_data'],
                    encrypted_data['salt'],
                    output_path
                ) 