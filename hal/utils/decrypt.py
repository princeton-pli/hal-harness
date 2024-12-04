import os
import click
import json
from zipfile import ZipFile
from pathlib import Path
from cryptography.fernet import Fernet
from typing import Optional
from ..utils.logging_utils import (
    print_step, 
    print_success, 
    print_error, 
    print_header,
    create_progress,
    console
)
from rich.table import Table
from rich.box import ROUNDED
from dotenv import load_dotenv
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from .json_encryption import JsonEncryption

load_dotenv()

# Replace the hardcoded key with proper key generation/loading
def get_encryption_key() -> bytes:
    """Get the encryption key in Fernet-compatible format."""
    base_key = "hal1234".ljust(32, '0')  # Pad with zeros to make it 32 bytes
    return base64.urlsafe_b64encode(base_key.encode())

def decrypt_json(encrypted_data: str, salt: str) -> dict:
    """Decrypt JSON data.
    
    Args:
        encrypted_data (str): Base64 encoded encrypted data
        salt (str): Base64 encoded salt used for encryption
        
    Returns:
        dict: Decrypted JSON data
    """
    try:
        # Decode salt and create cipher
        salt_bytes = base64.b64decode(salt.encode('utf-8'))
        cipher = JsonEncryption("hal1234", salt=salt_bytes)
        
        # Decode the encrypted data from base64
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        
        # Decrypt the data
        decrypted_data = cipher.cipher.decrypt(encrypted_bytes)
        
        # Parse and return the JSON
        return json.loads(decrypted_data.decode('utf-8'))
    except Exception as e:
        raise ValueError(f"Decryption failed: {str(e)}")

def decrypt_file(encrypted_file_path: Path, progress=None, task=None) -> None:
    """
    Decrypt and unzip a single file that was encrypted during upload.
    
    Args:
        encrypted_file_path: Path to the encrypted file
        progress: Optional progress bar
        task: Optional task ID for progress tracking
    """
    try:
        if progress:
            progress.update(task, description=f"Decrypting {encrypted_file_path.name}")
            
        # Read and unzip the file first
        if progress:
            progress.update(task, description=f"Extracting zip {encrypted_file_path.name}")
            
        with ZipFile(encrypted_file_path, 'r') as zip_file:
            # Get the encrypted JSON file from the zip
            json_filename = zip_file.namelist()[0]
            with zip_file.open(json_filename) as f:
                encrypted_data = json.load(f)
        
        # Decrypt the JSON data
        if progress:
            progress.update(task, description=f"Decrypting JSON content")
            
        decrypted_json = decrypt_json(
            encrypted_data['encrypted_data'],
            encrypted_data['salt']
        )
        
        # Write to output file
        output_path = encrypted_file_path.with_suffix('.json')
        if progress:
            progress.update(task, description=f"Writing {output_path.name}")
        with open(output_path, 'w') as f:
            json.dump(decrypted_json, f, indent=2)
            
        if progress:
            progress.update(task, advance=1)
            
        # Create summary table
        table = Table(title=f"Decryption Summary for {encrypted_file_path.name}", box=ROUNDED)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Input File", str(encrypted_file_path))
        table.add_row("Output File", str(output_path))
        table.add_row("Status", "✓ Success")
        
        console.print(table)
        
    except Exception as e:
        print_error(f"Error decrypting {encrypted_file_path}: {str(e)}")
        if progress:
            progress.update(task, description=f"Failed: {encrypted_file_path.name}")
        raise

def decrypt_directory(directory_path: Path) -> None:
    """
    Decrypt all encrypted files in a directory.
    
    Args:
        directory_path: Path to directory containing encrypted files
    """
    # Find all potential encrypted files
    encrypted_files = list(directory_path.glob("*.zip"))
    
    if not encrypted_files:
        print_error(f"No encrypted files found in {directory_path}")
        return
        
    print_step(f"Found {len(encrypted_files)} encrypted files")
    
    # Create progress bar
    with create_progress() as progress:
        # Add overall progress task
        main_task = progress.add_task(
            "Decrypting files...",
            total=len(encrypted_files)
        )
        
        # Process each file
        for file_path in encrypted_files:
            try:
                decrypt_file(file_path, progress, main_task)
            except Exception as e:
                print_error(f"Failed to decrypt {file_path}: {e}")
                continue
        
        # Ensure progress bar completes
        progress.update(main_task, completed=len(encrypted_files))

@click.command()
@click.option('-F', '--file', 'file_path', type=click.Path(exists=True), help='Path to encrypted file')
@click.option('-D', '--directory', 'directory_path', type=click.Path(exists=True), help='Path to directory containing encrypted files')
def decrypt_cli(file_path: Optional[str], directory_path: Optional[str]) -> None:
    """Decrypt files that were encrypted during upload to HAL."""
    print_header("HAL Decrypt")
    
    if not file_path and not directory_path:
        print_error("Please provide either a file (-F) or directory (-D) to decrypt")
        return
        
    if file_path and directory_path:
        print_error("Please provide either a file (-F) or directory (-D), not both")
        return
        
    try:
        if file_path:
            print_step(f"Decrypting single file: {file_path}")
            with create_progress() as progress:
                task = progress.add_task("Decrypting...", total=1)
                decrypt_file(Path(file_path), progress, task)
        else:
            print_step(f"Decrypting all files in directory: {directory_path}")
            decrypt_directory(Path(directory_path))
            
        print_success("Decryption completed successfully")
            
    except Exception as e:
        print_error(f"Decryption failed: {str(e)}")
        return

if __name__ == "__main__":
    decrypt_cli() 