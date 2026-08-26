from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from wenmai.factories.loader import ensure_providers
from wenmai.mcp.ask import ask_wenmai


def create_mcp_server() -> MCPServer:
    ensure_providers()
    server = MCPServer(
        "wenmai",
        instructions="Query the 福云·文脉助手 knowledge base with natural-language questions.",
    )

    @server.tool(
        name="ask_wenmai",
        description="Ask the Minpai culture knowledge base and get an answer with citations.",
    )
    def ask_wenmai_tool(
        question: str,
        culture_domain: str | None = None,
    ) -> dict[str, object]:
        return ask_wenmai(question, culture_domain=culture_domain)

    return server
