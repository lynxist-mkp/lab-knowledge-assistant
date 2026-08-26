#!/usr/bin/env python3
"""Start the 福云·文脉助手 MCP server on stdio transport."""

from __future__ import annotations

from wenmai.mcp.server import create_mcp_server


def main() -> None:
    server = create_mcp_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
