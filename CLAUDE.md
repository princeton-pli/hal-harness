Always make the smallest possible change to satisfy the request.
Keep the code DRY and follow best-practices.
This code is built to last a long time; prioritize longevity and regularity over experiments and hacks.
Keep in mind YAGNI; don't build anything you don't need.

<!-- claude-reliability:binary-instructions managed section - DO NOT EDIT -->
## claude-reliability Binary

The `claude-reliability` binary for this project is located at:

    .claude-reliability/bin/claude-reliability

Always use this path when running commands. Do NOT use bare `claude-reliability`,
do NOT use paths containing `~/.claude-reliability/`, and do NOT use `$PLUGIN_ROOT_DIR`
or any other variable to construct the path.

Example usage:

    .claude-reliability/bin/claude-reliability work list
    .claude-reliability/bin/claude-reliability work next
    .claude-reliability/bin/claude-reliability work on <id>
    .claude-reliability/bin/claude-reliability work update <id> --status complete
<!-- end claude-reliability:binary-instructions -->
