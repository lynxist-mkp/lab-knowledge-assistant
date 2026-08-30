from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from wenmai.config import Settings
from wenmai.mcp.summary import GetDocumentSummaryError, get_document_summary
from wenmai.pipelines.query import QueryGenerationError, ask_question
from wenmai.runtime import create_runtime


def create_mcp_server(settings: Settings | None = None) -> MCPServer:
    runtime = create_runtime(settings)
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
        retrieval_mode: str | None = None,
        rerank_enabled: bool | None = None,
    ) -> dict[str, object]:
        try:
            result = ask_question(
                question,
                runtime.settings,
                culture_domain=culture_domain,
                retrieval_mode=retrieval_mode,
                rerank_enabled=rerank_enabled,
                knowledge=runtime.knowledge,
            )
            return result.as_dict()
        except QueryGenerationError as exc:
            raise RuntimeError(f"{exc} (trace_id={exc.trace_id})") from exc

    @server.tool(
        name="get_document_summary",
        description=(
            "Fetch a document card by document_id: title, culture domain, "
            "enricher summary, and chunk count."
        ),
    )
    def get_document_summary_tool(document_id: str) -> dict[str, object]:
        try:
            return get_document_summary(
                document_id,
                runtime.settings,
                knowledge=runtime.knowledge,
            )
        except GetDocumentSummaryError as exc:
            raise RuntimeError(str(exc)) from exc

    return server
