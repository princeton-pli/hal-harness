# Build docker image

```
docker buildx build --platform linux/amd64 --load -t hal-core-agent-docker:latest .
```

Once the image is built, this can be run inside the harness using:

```
./run_docker_on_vm.py \
  --image hal-core-agent-docker:latest \
  ... (other flags)
```
