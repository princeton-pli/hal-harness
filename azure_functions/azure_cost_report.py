"""Daily Azure cost report grouped by ExecutedBy tag, posted to Slack."""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential
from azure.mgmt.costmanagement import CostManagementClient

logger = logging.getLogger(__name__)


def get_yesterday_costs(subscription_id: str, resource_group: str | None = None) -> dict:
    """Query Azure Cost Management for yesterday's costs grouped by ExecutedBy tag.

    Returns:
        dict with keys: rows (list of [cost, currency, user]), total, currency, date
    """
    credential = DefaultAzureCredential()
    client = CostManagementClient(credential)

    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    time_period = {
        "from": datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=timezone.utc),
        "to": datetime(yesterday.year, yesterday.month, yesterday.day, 23, 59, 59, tzinfo=timezone.utc),
    }

    if resource_group:
        scope = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}"
    else:
        scope = f"/subscriptions/{subscription_id}"

    query = {
        "type": "ActualCost",
        "timeframe": "Custom",
        "time_period": time_period,
        "dataset": {
            "granularity": "None",
            "aggregation": {"totalCost": {"name": "Cost", "function": "Sum"}},
            "grouping": [{"type": "TagKey", "name": "ExecutedBy"}],
        },
    }

    # Azure Cost Management allows ~4 calls/min/scope, shared across tenant
    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = client.query.usage(scope=scope, parameters=query)
            break
        except HttpResponseError as e:
            if e.status_code == 429 and attempt < max_retries - 1:
                retry_after = int(e.response.headers.get("Retry-After", 0))
                wait = max(retry_after, 30 * (2 ** attempt))
                logger.warning(f"Rate limited, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise

    # Response columns: Cost, TagKey, TagValue, Currency
    rows = []
    total = 0.0
    currency = "USD"
    for row in result.rows:
        cost = float(row[0])
        user = row[2] if row[2] else "untagged"
        cur = row[3]
        currency = cur
        total += cost
        rows.append({"user": user, "cost": cost, "currency": cur})

    rows.sort(key=lambda r: r["cost"], reverse=True)

    return {
        "rows": rows,
        "total": total,
        "currency": currency,
        "date": str(yesterday),
        "resource_group": resource_group,
    }


def format_slack_message(data: dict) -> dict:
    """Format cost data as a Slack message payload."""
    date = data["date"]
    currency = data["currency"]
    rg = data.get("resource_group")

    lines = [f"*Azure Daily Cost Report — {date}*\n"]
    lines.append(f"{'User':<16} {'Cost (' + currency + ')':>12}")
    lines.append("-" * 30)

    for row in data["rows"]:
        lines.append(f"{row['user']:<16} ${row['cost']:>10.2f}")

    lines.append("-" * 30)
    lines.append(f"{'Total':<16} ${data['total']:>10.2f}")

    if rg:
        lines.append(f"\nResource Group: {rg}")

    return {"text": "\n".join(lines)}


def post_to_slack(webhook_url: str, message: dict) -> None:
    """Post a message to Slack via incoming webhook."""
    resp = requests.post(webhook_url, json=message, timeout=10)
    resp.raise_for_status()
    logger.info("Slack message posted successfully")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("azure").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Post daily Azure cost report to Slack")
    parser.add_argument(
        "--resource-group",
        default=os.getenv("AZURE_RESOURCE_GROUP_NAME"),
        help="Azure resource group to scope costs to (default: AZURE_RESOURCE_GROUP_NAME env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the report to stdout instead of posting to Slack",
    )
    args = parser.parse_args()

    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    if not subscription_id:
        logger.error("AZURE_SUBSCRIPTION_ID environment variable is required")
        sys.exit(1)

    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url and not args.dry_run:
        logger.error("SLACK_WEBHOOK_URL environment variable is required (or use --dry-run)")
        sys.exit(1)

    try:
        data = get_yesterday_costs(subscription_id, args.resource_group)
        message = format_slack_message(data)
    except Exception as e:
        error_msg = f":x: *Azure Cost Report Failed*\n\n`{type(e).__name__}: {e}`"
        logger.error(error_msg)
        if webhook_url and not args.dry_run:
            post_to_slack(webhook_url, {"text": error_msg})
        sys.exit(1)

    if args.dry_run:
        print(message["text"])
    else:
        post_to_slack(webhook_url, message)


if __name__ == "__main__":
    main()
