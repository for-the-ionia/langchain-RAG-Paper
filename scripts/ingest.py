from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from app.data_loader import load_excel_documents
from app.rag import build_simple_index, create_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Excel paper records into the local JSON vector index")
    parser.add_argument("--excel", required=True, help="Excel file path")
    parser.add_argument("--recreate", action="store_true", help="Accepted for compatibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    documents = load_excel_documents(args.excel)
    embeddings = create_embeddings(settings)
    build_simple_index(documents, embeddings, settings.simple_index_path)

    print(f"Imported papers: {len(documents)}")
    print(f"Index path: {Path(settings.simple_index_path)}")
    print(f"LLM provider: {settings.provider}")
    print(f"Embedding provider: {settings.embeddings_provider}")


if __name__ == "__main__":
    main()
