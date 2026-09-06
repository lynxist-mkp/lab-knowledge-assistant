#!/usr/bin/env python3
"""Start the 课题组知识助手 MCP server on stdio transport."""

from __future__ import annotations

from lab_knowledge.mcp.server import create_mcp_server
from lab_knowledge.mcp.stdio_safety import install_mcp_stdio_discipline


def main() -> None:
    install_mcp_stdio_discipline()
    server = create_mcp_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
