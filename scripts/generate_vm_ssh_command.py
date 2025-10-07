"""Utility to print the SSH command for a running HAL VM.

Usage:
    python scripts/generate_vm_ssh_command.py --vm-name agent-gaia-...

The script uses the same Azure configuration as the harness. It looks up the
VM's public IP and prints the `ssh` command, defaulting to the `agent` user and
the private key specified via --key-path or the SSH_PRIVATE_KEY_PATH env var.
"""

from __future__ import annotations

import argparse
import os

from pathlib import Path

from dotenv import load_dotenv

from hal.utils.azure_utils import VirtualMachineManager


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate SSH command for an Azure VM")
    parser.add_argument("--vm-name", required=True, help="Name of the VM (e.g. agent-gaia-xxxx)")
    parser.add_argument(
        "--username",
        default="agent",
        help="SSH username (defaults to agent)",
    )
    parser.add_argument(
        "--key-path",
        default=None,
        help="Path to the SSH private key. Defaults to $SSH_PRIVATE_KEY_PATH",
    )
    parser.add_argument(
        "--subscription-id",
        default=None,
        help="Override AZURE_SUBSCRIPTION_ID for this call",
    )
    parser.add_argument(
        "--resource-group",
        default=None,
        help="Override AZURE_RESOURCE_GROUP_NAME for this call",
    )
    parser.add_argument(
        "--location",
        default=None,
        help="Override AZURE_LOCATION for this call",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Load project root .env if present
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    key_path = args.key_path or os.getenv("SSH_PRIVATE_KEY_PATH")
    if not key_path:
        raise SystemExit("Provide --key-path or set SSH_PRIVATE_KEY_PATH.")

    if args.subscription_id:
        os.environ["AZURE_SUBSCRIPTION_ID"] = args.subscription_id
    if args.resource_group:
        os.environ["AZURE_RESOURCE_GROUP_NAME"] = args.resource_group
    if args.location:
        os.environ["AZURE_LOCATION"] = args.location

    missing = [
        var
        for var in ("AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP_NAME", "AZURE_LOCATION")
        if not os.getenv(var)
    ]
    if missing:
        formatted = ", ".join(missing)
        raise SystemExit(f"Missing required Azure env vars: {formatted}. Provide them via --options or environment.")

    manager = VirtualMachineManager()

    if not manager.resource_group_name:
        raise SystemExit("AZURE_RESOURCE_GROUP_NAME is not set.")

    public_ip_name = f"{args.vm_name}-public-ip"
    public_ip = manager.network_client.public_ip_addresses.get(
        manager.resource_group_name, public_ip_name
    ).ip_address

    command = f"ssh -i {key_path} {args.username}@{public_ip}"
    print(command)


if __name__ == "__main__":
    main()
