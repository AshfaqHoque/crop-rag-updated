import asyncio
import json
import sqlite3
from types import SimpleNamespace

from langchain_core.documents import Document

from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.pipeline.nodes import generate_sql_answer as generate_sql_answer_module
from app.services.pipeline.nodes import resolve_entities as resolve_entities_module
from app.services.pipeline.nodes import sql_generate as sql_generate_module
from app.services.sql.create_crop_db import create_crops_db


def _registry_payload(image):
    return {
        "data": {
            "getAllCropsFullDetails": {
                "rows": [
                    {
                        "id": 1,
                        "crop_name": "Rice",
                        "crop_bangla_name": "ধান",
                        "pesticide": [
                            {
                                "id": 10,
                                "crop_id": 1,
                                "disease_name": "Stem borer",
                                "disease_type": "insect",
                                "image": image,
                                "chemical": [],
                            }
                        ],
                    }
                ]
            }
        }
    }


def test_create_crops_db_migrates_and_updates_pest_images(tmp_path):
    registry = tmp_path / "crops.json"
    database = tmp_path / "crops.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE pesticide (
                id INTEGER PRIMARY KEY,
                disease_name TEXT,
                disease_type TEXT,
                crop_id INTEGER
            )
            """
        )
        connection.execute(
            "INSERT INTO pesticide VALUES (10, 'Old name', 'insect', 1)"
        )

    registry.write_text(
        json.dumps(
            _registry_payload(
                [{"image": "https://example.test/borer-1.jpg"}, {"image": "https://example.test/borer-2.jpg"}]
            )
        ),
        encoding="utf-8",
    )
    create_crops_db(registry, database)

    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(pesticide)")}
        row = connection.execute(
            "SELECT disease_name, image_paths FROM pesticide WHERE id = 10"
        ).fetchone()
    assert "image_paths" in columns
    assert row == (
        "Stem borer",
        json.dumps(
            ["https://example.test/borer-1.jpg", "https://example.test/borer-2.jpg"],
            ensure_ascii=False,
        ),
    )

    registry.write_text(
        json.dumps(_registry_payload([{"image": "https://example.test/updated.jpg"}])),
        encoding="utf-8",
    )
    create_crops_db(registry, database)
    with sqlite3.connect(database) as connection:
        updated_paths = connection.execute(
            "SELECT image_paths FROM pesticide WHERE id = 10"
        ).fetchone()[0]
    assert json.loads(updated_paths) == ["https://example.test/updated.jpg"]


def test_pest_entity_resolution_uses_disease_index(monkeypatch):
    class Store:
        def similarity_search_with_score(self, mention, k, filter):
            assert mention == "stem borer"
            assert filter == {"entity_type": "disease"}
            return [(Document(page_content="Stem borer", metadata={"canonical_name": "Stem borer"}), 0.1)]

    monkeypatch.setattr(resolve_entities_module, "get_vector_store", lambda collection: Store())
    result = resolve_entities_module.resolve_entities(
        {
            "query_plan": {
                "operation": "lookup",
                "entities_to_resolve": [{"type": "pest", "mention": "stem borer"}],
            }
        }
    )
    assert result["query_plan"]["entities_to_resolve"] == [
        {"type": "pest", "mention": "Stem borer"}
    ]


def test_sql_generation_requests_images_for_pest_results(monkeypatch):
    class SQLResult:
        sql = "SELECT disease_name, image_paths FROM pesticide"

    captured = {}

    def invoke_structured(result_type, messages, temperature):
        captured["prompt"] = "\n".join(message.content for message in messages)
        return SQLResult()

    monkeypatch.setattr(sql_generate_module, "invoke_structured", invoke_structured)
    result = sql_generate_module.sql_generate(
        {
            "rewritten_query": "Show me stem borer",
            "query_plan": {
                "operation": "lookup",
                "target": "pest",
                "entities_to_resolve": [{"type": "pest", "mention": "Stem borer"}],
            },
        }
    )
    assert result["sql_query"] == SQLResult.sql
    assert "image_paths" in captured["prompt"]
    assert "For pest/disease lookup or list results" in captured["prompt"]


def test_sql_answer_renders_all_unique_pest_images(monkeypatch):
    monkeypatch.setattr(
        generate_sql_answer_module,
        "invoke_text",
        lambda messages: "Stem borer can damage the crop.",
    )
    image_paths = [
        "https://example.test/borer one.jpg",
        "https://example.test/borer-two.jpg",
    ]
    result = generate_sql_answer_module.generate_sql_answer(
        {
            "raw_query": "Show me stem borer",
            "rewritten_query": "Show me stem borer",
            "language_type": "en",
            "sql_results": [
                {"disease_name": "Stem borer", "image_paths": json.dumps(image_paths)},
                {"disease_name": "Stem borer", "image_paths": json.dumps([image_paths[0]])},
            ],
        }
    )
    assert result["answer"] == (
        "Stem borer can damage the crop.\n\n"
        "![Pest image](https://example.test/borer%20one.jpg)\n"
        "![Pest image](https://example.test/borer-two.jpg)"
    )
    assert result["messages"][0].content == result["answer"]


def test_streamed_sql_answer_keeps_rendered_images():
    class SQLAnswerGraph:
        async def astream(self, state, config, stream_mode):
            yield "messages", (
                SimpleNamespace(id="answer", content="Stem borer affects rice."),
                {"langgraph_node": "generate_sql_answer"},
            )
            yield "updates", {
                "generate_sql_answer": {
                    "answer": "Stem borer affects rice.\n\n![Pest image](https://example.test/borer.jpg)"
                }
            }

    service = ChatService(graph=SQLAnswerGraph())
    async def collect_events():
        chunks = service.stream_chat(
            ChatRequest(session_id="session", message="Show stem borer", language_type="en")
        )
        return [json.loads(event.removeprefix("data: ").strip()) async for event in chunks]

    events = asyncio.run(collect_events())
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"].endswith("![Pest image](https://example.test/borer.jpg)")
