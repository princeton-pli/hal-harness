# CRUX Agent Docker

This is a WIP to create a claude-in-a-loop container that we can run for long-running tasks.

## Development

```
docker run --rm \
    -v /tmp/docker_run/test-planning:/workspace/logs \
    -v /tmp/docker_run/test-planning/code:/workspace/code \
    -v /tmp/docker_run/test-planning/results:/workspace/results \
    -e HAL_RUN_ID=test-planning-123 \
    -e HAL_TASK_ID=test-task-456 \
    -e ANTHROPIC_API_KEY="APIKEYAPIKEYAPIKEYAPIKEYAPIKEY" \
    -e PROJECT_DESCRIPTION="An AI agent builds a simple mobile app from scratch and publishes it on the Apple App Store. The agent is responsible for every step of the process except those where human involvement is required by policy: writing the code, building the app, preparing metadata, drafting and hosting a privacy policy, submitting for review, and handling any rejection feedback. Get as far as you can without an Apple ID or pre-paid debit card and save a file REQUIREMENTS.md at the end that includes specific requirements of things that you would need to fully get this live (Apple ID, a VM, an email account, pre-paid debit card, etc.) - Success is binary: the app is publicly available for download, or it is not." \
    crux-agent:test
```
