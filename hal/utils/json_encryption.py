import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class JsonEncryption:
    def __init__(self, password: str, salt: bytes = None):
        """Initialize encryption with a password.
        
        Args:
            password (str): The password used for encryption/decryption
            salt (bytes): The salt used for key derivation
        """
        # Store the original password
        self._original_password = password
        # Generate a salt for key derivation or use provided salt
        self.salt = salt if salt is not None else os.urandom(16)
        # Create a key from the password
        self.key = self._generate_key(password.encode(), self.salt)
        # Initialize Fernet cipher
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
    
    def encrypt_json(self, json_data: dict) -> dict:
        """Encrypt JSON data.
        
        Args:
            json_data (dict): JSON data to encrypt
            
        Returns:
            dict: Dictionary containing the encrypted data and salt
        """
        # Convert JSON to string
        json_str = json.dumps(json_data)
        
        # Encrypt the JSON string
        encrypted_data = self.cipher.encrypt(json_str.encode())
        
        # Return encrypted data and salt
        return {
            'encrypted_data': base64.b64encode(encrypted_data).decode('utf-8'),
            'salt': base64.b64encode(self.salt).decode('utf-8')
        }
    
    def decrypt_json(self, encrypted_data: str, salt: str) -> dict:
        """Decrypt JSON data.
        
        Args:
            encrypted_data (str): Base64 encoded encrypted data
            salt (str): Base64 encoded salt used for encryption
            
        Returns:
            dict: Decrypted JSON data
        """
        try:
            # Decode the encrypted data and salt from base64
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            
            # Decrypt the data
            decrypted_data = self.cipher.decrypt(encrypted_bytes)
            
            # Parse and return the JSON
            return json.loads(decrypted_data.decode('utf-8'))
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")

    def encrypt_json_file(self, input_path: str, output_path: str):
        """Encrypt JSON file.
        
        Args:
            input_path (str): Path to input JSON file
            output_path (str): Path to save encrypted file
        """
        try:
            # Read JSON file
            with open(input_path, 'r') as file:
                json_data = json.load(file)
            
            # Encrypt the data
            encrypted = self.encrypt_json(json_data)
            
            # Write encrypted data to file
            with open(output_path, 'w') as file:
                json.dump(encrypted, file, indent=2)
                
        except Exception as e:
            raise ValueError(f"File encryption failed: {str(e)}")

    def decrypt_json_file(self, input_path: str, output_path: str):
        """Decrypt JSON file.
        
        Args:
            input_path (str): Path to encrypted JSON file
            output_path (str): Path to save decrypted file
        """
        try:
            # Read encrypted file
            with open(input_path, 'r') as file:
                encrypted_data = json.load(file)
            
            # Create new cipher with the stored salt
            salt = base64.b64decode(encrypted_data['salt'].encode('utf-8'))
            # Fix: Pass the original password instead of the key
            self.__init__(password=self._original_password, salt=salt)  # Reinitialize with correct salt
            
            # Decrypt the data
            decrypted = self.decrypt_json(
                encrypted_data['encrypted_data'],
                encrypted_data['salt']
            )
            
            # Write decrypted data to file
            with open(output_path, 'w') as file:
                json.dump(decrypted, file, indent=2)
                
        except Exception as e:
            raise ValueError(f"File decryption failed: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Example JSON data
    sample_data = {
        "username": "john_doe",
        "email": "john@example.com",
        "settings": {
            "theme": "dark",
            "notifications": True
        }
    }
    
    try:
        # Initialize encryptor with a password
        encryptor = JsonEncryption("your-secure-password")
        
        # Example 1: Encrypt and decrypt data in memory
        print("Example 1: In-memory encryption/decryption")
        encrypted = encryptor.encrypt_json(sample_data)
        print(f"Encrypted data: {encrypted}")
        
        decrypted = encryptor.decrypt_json(
            encrypted['encrypted_data'],
            encrypted['salt']
        )
        print(f"Decrypted data: {decrypted}")
        
        # Example 2: Encrypt and decrypt files
        print("\nExample 2: File encryption/decryption")
        encryptor.encrypt_json_file("input.json", "encrypted.json")
        print("File encrypted successfully")
        
        encryptor.decrypt_json_file("encrypted.json", "decrypted.json")
        print("File decrypted successfully")
        
    except Exception as e:
        print(f"Error: {str(e)}")