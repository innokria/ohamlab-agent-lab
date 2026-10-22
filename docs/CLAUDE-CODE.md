# Claude Code setup checklist

1. Confirm the OhamLab MCP endpoint or local server command is working.
2. Use Claude Code's current MCP configuration command/documentation to add the server.
3. For a hosted server, use the exact Streamable HTTP URL and authentication format supported by your installed Claude Code version.
4. For a local server, configure the actual executable and arguments documented by the MCP package.
5. Restart/reload Claude Code if required, then ask it to list or use the connected MCP tools.
6. Do not paste secrets into committed project files.

The Claude Code MCP config is distinct from `config.yaml`, which configures the Agent Lab as an MCP client.
