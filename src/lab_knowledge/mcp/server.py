from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from lab_knowledge.config import Settings
from lab_knowledge.http.ask_governor import AskSaturationError
from lab_knowledge.knowledge.document_management import create_document_management
from lab_knowledge.mcp.errors import AskSaturationToolError
from lab_knowledge.mcp.stdio_safety import redact_error_message
from lab_knowledge.mcp.tools.ask import AskAnswerError, ask_answer
from lab_knowledge.mcp.tools.collections import collections_get_stats, collections_list
from lab_knowledge.mcp.tools.documents import documents_delete, documents_get, documents_list
from lab_knowledge.mcp.tools.images import images_get_content, images_get_ref
from lab_knowledge.mcp.tools.reviews import reviews_approve, reviews_list_pending, reviews_reject
from lab_knowledge.runtime import create_runtime


def _mcp_error(exc: Exception) -> RuntimeError:
    return RuntimeError(redact_error_message(str(exc)))


def _ask_tool_error(exc: Exception) -> Exception:
    if isinstance(exc, AskAnswerError):
        return RuntimeError(
            redact_error_message(f"{exc} (trace_id={exc.trace_id})")
        )
    if isinstance(exc, AskSaturationError):
        return AskSaturationToolError(exc)
    return _mcp_error(exc)


def create_mcp_server(settings: Settings | None = None) -> MCPServer:
    runtime = create_runtime(settings)
    document_management = create_document_management(
        runtime.settings,
        knowledge=runtime.knowledge,
    )
    server = MCPServer(
        "lab_knowledge",
        instructions=(
            "Query and manage the 课题组知识助手 knowledge base. "
            "Use ask.answer or ask_lab_knowledge for questions; "
            "documents.* and reviews.* for admin. "
            "ask_wenmai remains available as a legacy alias."
        ),
    )

    def _ask_tool(
        question: str,
        collection_id: str | None = None,
        culture_domain: str | None = None,
        retrieval_mode: str | None = None,
        rerank_enabled: bool | None = None,
    ) -> dict[str, object]:
        return ask_answer(
            question,
            runtime.settings,
            collection_id=collection_id,
            culture_domain=culture_domain,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            knowledge=runtime.knowledge,
        )

    def _legacy_ask_tool(
        question: str,
        collection_id: str | None = None,
        culture_domain: str | None = None,
        retrieval_mode: str | None = None,
        rerank_enabled: bool | None = None,
    ) -> dict[str, object]:
        return _ask_tool(
            question,
            collection_id=collection_id,
            culture_domain=culture_domain,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
        )["data"]

    @server.tool(
        name="ask.answer",
        description="Ask the knowledge base and get an answer with citations.",
    )
    def ask_answer_tool(
        question: str,
        collection_id: str | None = None,
        culture_domain: str | None = None,
        retrieval_mode: str | None = None,
        rerank_enabled: bool | None = None,
    ) -> dict[str, object]:
        try:
            return _ask_tool(
                question,
                collection_id=collection_id,
                culture_domain=culture_domain,
                retrieval_mode=retrieval_mode,
                rerank_enabled=rerank_enabled,
            )
        except ValueError as exc:
            raise _mcp_error(exc) from exc
        except (AskAnswerError, AskSaturationError) as exc:
            raise _ask_tool_error(exc) from exc

    @server.tool(
        name="ask_lab_knowledge",
        description="Primary ask alias: ask the knowledge base and get an answer with citations.",
    )
    def ask_lab_knowledge_tool(
        question: str,
        collection_id: str | None = None,
        culture_domain: str | None = None,
        retrieval_mode: str | None = None,
        rerank_enabled: bool | None = None,
    ) -> dict[str, object]:
        try:
            return _legacy_ask_tool(
                question,
                collection_id=collection_id,
                culture_domain=culture_domain,
                retrieval_mode=retrieval_mode,
                rerank_enabled=rerank_enabled,
            )
        except ValueError as exc:
            raise _mcp_error(exc) from exc
        except (AskAnswerError, AskSaturationError) as exc:
            raise _ask_tool_error(exc) from exc

    @server.tool(
        name="ask_wenmai",
        description="Legacy ask alias: ask the knowledge base and get an answer with citations.",
    )
    def ask_wenmai_tool(
        question: str,
        collection_id: str | None = None,
        culture_domain: str | None = None,
        retrieval_mode: str | None = None,
        rerank_enabled: bool | None = None,
    ) -> dict[str, object]:
        try:
            return _legacy_ask_tool(
                question,
                collection_id=collection_id,
                culture_domain=culture_domain,
                retrieval_mode=retrieval_mode,
                rerank_enabled=rerank_enabled,
            )
        except ValueError as exc:
            raise _mcp_error(exc) from exc
        except (AskAnswerError, AskSaturationError) as exc:
            raise _ask_tool_error(exc) from exc

    @server.tool(
        name="collections.list",
        description="List configured knowledge collections with status-layered stats.",
    )
    def collections_list_tool() -> dict[str, object]:
        return collections_list(document_management)

    @server.tool(
        name="collections.get_stats",
        description="Get collection-level document and review-status statistics.",
    )
    def collections_get_stats_tool(collection_id: str | None = None) -> dict[str, object]:
        try:
            return collections_get_stats(document_management, collection_id=collection_id)
        except ValueError as exc:
            raise _mcp_error(exc) from exc

    @server.tool(
        name="documents.list",
        description="List documents in a collection, optionally filtered by culture domain.",
    )
    def documents_list_tool(
        collection_id: str | None = None,
        culture_domain: str | None = None,
    ) -> dict[str, object]:
        try:
            return documents_list(
                document_management,
                collection_id=collection_id,
                culture_domain=culture_domain,
            )
        except ValueError as exc:
            raise _mcp_error(exc) from exc

    @server.tool(
        name="documents.get",
        description="Get document detail card and image references by document_id.",
    )
    def documents_get_tool(
        document_id: str,
        collection_id: str | None = None,
    ) -> dict[str, object]:
        try:
            return documents_get(
                document_management,
                document_id,
                collection_id=collection_id,
            )
        except ValueError as exc:
            raise _mcp_error(exc) from exc

    @server.tool(
        name="documents.delete",
        description="Delete a document and its chunks, indexes, and images.",
    )
    def documents_delete_tool(
        document_id: str,
        collection_id: str | None = None,
    ) -> dict[str, object]:
        try:
            return documents_delete(
                document_management,
                document_id,
                collection_id=collection_id,
            )
        except ValueError as exc:
            raise _mcp_error(exc) from exc

    @server.tool(
        name="reviews.list_pending",
        description="List documents with pending-review chunks in a collection.",
    )
    def reviews_list_pending_tool(collection_id: str | None = None) -> dict[str, object]:
        try:
            return reviews_list_pending(document_management, collection_id=collection_id)
        except ValueError as exc:
            raise _mcp_error(exc) from exc

    @server.tool(
        name="reviews.approve",
        description="Approve a pending-review document for retrieval.",
    )
    def reviews_approve_tool(
        document_id: str,
        collection_id: str | None = None,
    ) -> dict[str, object]:
        try:
            return reviews_approve(
                document_management,
                document_id,
                collection_id=collection_id,
            )
        except ValueError as exc:
            raise _mcp_error(exc) from exc

    @server.tool(
        name="reviews.reject",
        description="Reject a pending-review document and remove it from the knowledge base.",
    )
    def reviews_reject_tool(
        document_id: str,
        collection_id: str | None = None,
    ) -> dict[str, object]:
        try:
            return reviews_reject(
                document_management,
                document_id,
                collection_id=collection_id,
            )
        except ValueError as exc:
            raise _mcp_error(exc) from exc

    @server.tool(
        name="images.get_ref",
        description="Get image reference metadata without inline bytes.",
    )
    def images_get_ref_tool(
        image_id: str,
        collection_id: str | None = None,
    ) -> dict[str, object]:
        try:
            return images_get_ref(
                document_management,
                image_id,
                collection_id=collection_id,
            )
        except ValueError as exc:
            raise _mcp_error(exc) from exc

    @server.tool(
        name="images.get_content",
        description="Opt-in: fetch image bytes as base64 with MIME type; storage paths are hidden.",
    )
    def images_get_content_tool(
        image_id: str,
        collection_id: str | None = None,
    ) -> dict[str, object]:
        try:
            return images_get_content(
                document_management,
                image_id,
                collection_id=collection_id,
            )
        except ValueError as exc:
            raise _mcp_error(exc) from exc

    return server
