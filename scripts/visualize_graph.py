"""Export the compiled LangGraph pipeline as an image."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.services.pipeline.graph import get_chat_graph


graph = get_chat_graph().get_graph()
png_path = ROOT / "graph.png"

png_path.write_bytes(graph.draw_mermaid_png())

print(f"Wrote {png_path}")