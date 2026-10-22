# DeepSeek Harness setup checklist

1. Confirm the deployed OhamLab MCP server's `/mcp/` endpoint and authentication work independently.
2. In DeepSeek Harness, configure an MCP server using the supported Streamable HTTP transport and exact deployed URL.
3. Configure the required authentication header through a secret/environment mechanism supported by your deployment.
4. Reload/restart the harness MCP client as required.
5. Verify `tools/list` returns the expected tools, then test a read-only search/fetch tool.
6. Confirm the model runtime actually receives tool schemas and dispatches tool calls; successful MCP initialization alone does not prove tool execution works.
7. Keep Agent Lab's `config.yaml` separate. This project is an MCP client, not an inbound MCP server for DSH to call.

Exact configuration syntax depends on the installed DeepSeek Harness version and deployment settings.
