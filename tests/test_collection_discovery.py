"""Contract tests for explicit collection discovery via settings registry."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lab_knowledge.config import CollectionRegistration, Settings
from lab_knowledge.knowledge import create_document_management, create_knowledge
from lab_knowledge.knowledge.collections import (
    CollectionReadModel,
    UnknownCollectionError,
)
from lab_knowledge.mcp.tools.collections import collections_get_stats, collections_list
from tests.conftest import register_collection


def test_list_collections_legacy_single_collection_config(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    model = CollectionReadModel(test_settings, knowledge)

    collections = model.list_collections()

    assert len(collections) == 1
    assert collections[0].collection_id == test_settings.product.collection
    assert collections[0].display_name == test_settings.product.name


def test_list_collections_includes_registered_empty_collection(test_settings: Settings) -> None:
    other_id = "empty-collection"
    settings = register_collection(test_settings, other_id, display_name="空集合")
    knowledge = create_knowledge(settings)
    model = CollectionReadModel(settings, knowledge)

    collections = model.list_collections()

    assert [item.collection_id for item in collections] == [
        settings.product.collection,
        other_id,
    ]
    assert collections[1].display_name == "空集合"
    assert collections[1].stats.document_count == 0
    assert collections[1].stats.chunk_count == 0


def test_registered_empty_collection_is_routable_with_zero_stats(
    test_settings: Settings,
) -> None:
    other_id = "empty-collection"
    settings = register_collection(test_settings, other_id)
    knowledge = create_knowledge(settings)
    model = CollectionReadModel(settings, knowledge)

    stats = model.get_stats(other_id)
    collection = model.get_collection(other_id)

    assert stats.document_count == 0
    assert stats.chunk_count == 0
    assert collection.collection_id == other_id


def test_document_management_registered_empty_collection(
    test_settings: Settings,
) -> None:
    other_id = "empty-collection"
    settings = register_collection(test_settings, other_id)
    mgmt = create_document_management(settings)

    scoped = mgmt.for_collection(other_id)
    stats = scoped.get_collection_stats()

    assert scoped.scope.collection_id == other_id
    assert stats.document_count == 0
    assert stats.chunk_count == 0
    assert scoped.list_documents() == []


def test_mcp_collections_registered_empty_collection(test_settings: Settings) -> None:
    other_id = "empty-collection"
    settings = register_collection(test_settings, other_id, display_name="MCP 空集合")
    mgmt = create_document_management(settings)

    listed = collections_list(mgmt)
    assert {item["collection_id"] for item in listed["data"]} == {
        settings.product.collection,
        other_id,
    }
    empty_entry = next(item for item in listed["data"] if item["collection_id"] == other_id)
    assert empty_entry["display_name"] == "MCP 空集合"
    assert empty_entry["stats"]["document_count"] == 0

    stats = collections_get_stats(mgmt, collection_id=other_id)
    assert stats["scope"]["collection_id"] == other_id
    assert stats["data"]["document_count"] == 0


def test_unregistered_collection_still_raises(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    model = CollectionReadModel(test_settings, knowledge)

    with pytest.raises(UnknownCollectionError, match="unknown collection: missing"):
        model.get_stats("missing")

    with pytest.raises(UnknownCollectionError, match="unknown collection: missing"):
        model.resolve_scope("missing")


def test_default_collection_display_name_override(test_settings: Settings) -> None:
    default_id = test_settings.product.collection
    settings = replace(
        test_settings,
        collections=[
            CollectionRegistration(
                collection_id=default_id,
                display_name="默认展示名",
            )
        ],
    )
    model = CollectionReadModel(settings, create_knowledge(settings))

    listed = model.list_collections()

    assert len(listed) == 1
    assert listed[0].display_name == "默认展示名"
