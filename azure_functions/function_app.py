"""Azure Function: Daily cost report posted to Slack at 8:00 AM Eastern."""

import logging
import os

import azure.functions as func

from azure_cost_report import get_yesterday_costs, format_slack_message, post_to_slack

app = func.FunctionApp()

# WEBSITE_TIME_ZONE must be set to "America/New_York" in app settings
# so this fires at 8:00 AM Eastern (auto-adjusts for DST)
@app.timer_trigger(schedule="0 0 8 * * *", arg_name="timer", run_on_startup=False)
def cost_report_timer(timer: func.TimerRequest) -> None:
    webhook_url = os.environ["SLACK_WEBHOOK_URL"]
    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    resource_group = os.getenv("AZURE_RESOURCE_GROUP_NAME")

    try:
        data = get_yesterday_costs(subscription_id, resource_group)
        message = format_slack_message(data)
        post_to_slack(webhook_url, message)
        logging.info("Cost report posted to Slack")
    except Exception as e:
        error_msg = f":x: *Azure Cost Report Failed*\n\n`{type(e).__name__}: {e}`"
        logging.error(error_msg)
        post_to_slack(webhook_url, {"text": error_msg})
