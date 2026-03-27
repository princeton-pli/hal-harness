# Docker Orchestrator

Runs `hal-eval` inside a Docker container.
`hal-eval` uses the standard local `Runner` inside the container.

## Build

```bash
docker build -t hal-agent orchestrators/docker/
```

## Run

```bash
docker run \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/agents/my_agent:/app/my_agent:ro \
    --env-file .env \
    hal-agent \
    hal-eval --agent_dir /app/my_agent --benchmark swebench_verified \
    --agent_name "my-agent" --agent_function agent.run
```
