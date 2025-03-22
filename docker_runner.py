import os

class DockerRunner:
    def __init__(self, docker_client):
        self.docker_client = docker_client

    def run_container(self, task_id, container_id, temp_dir, container_command):
        # Run container using Docker Python SDK
        print("TMP DIR CONTENTS 2: ", os.listdir(temp_dir))
        verbose_logger.debug(f"Running Docker container for task {task_id}")
        container = self.docker_client.containers.run(
            image=DOCKER_IMAGE_NAME,
            name=container_id,
            detach=True,  # Run in background
            working_dir="/workspace",
            volumes={temp_dir_abs: {'bind': '/workspace', 'mode': 'rw'}},
            mem_limit="4g",  # 4GB memory limit
            memswap_limit="4g",  # No swap
            command=["tail", "-f", "/dev/null"]  # Keep container running
        )
        
        # Now we can safely execute our command
        container.exec_run(["bash", "-c", container_command]) 