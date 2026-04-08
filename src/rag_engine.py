"""RAG engine for retrieving relevant codebase context during reviews."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_PERSIST_DIR, should_skip_file

logger = logging.getLogger(__name__)

# File extensions to index
INDEXABLE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
    ".kt", ".scala", ".sh", ".sql",
}


class RAGEngine:
    """Retrieval-Augmented Generation engine for codebase context.

    Indexes your codebase into a vector store and retrieves relevant code
    snippets during PR review to give the LLM full context.
    """

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\nclass ", "\ndef ", "\nfunction ", "\n\n", "\n", " "],
        )
        self._vectorstore: Chroma | None = None

    @property
    def vectorstore(self) -> Chroma:
        """Lazy-load the vector store."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name="codebase",
            )
        return self._vectorstore

    def index_directory(self, directory: str, file_filter: set[str] | None = None) -> int:
        """Index all source files in a directory into the vector store.

        Args:
            directory: Root directory to index.
            file_filter: Optional set of file extensions to include.

        Returns:
            Number of chunks indexed.
        """
        extensions = file_filter or INDEXABLE_EXTENSIONS
        root = Path(directory)
        documents = []

        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in extensions:
                continue
            if should_skip_file(str(file_path)):
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if not content.strip():
                    continue

                rel_path = str(file_path.relative_to(root))
                chunks = self.text_splitter.create_documents(
                    texts=[content],
                    metadatas=[{
                        "source": rel_path,
                        "language": file_path.suffix.lstrip("."),
                    }],
                )
                documents.extend(chunks)
            except Exception as e:
                logger.warning("Failed to index %s: %s", file_path, e)
                continue

        if not documents:
            logger.warning("No documents found to index in %s", directory)
            return 0

        # Batch insert into vector store
        batch_size = 500
        total = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self.vectorstore.add_documents(batch)
            total += len(batch)
            logger.info("Indexed %d/%d chunks", total, len(documents))

        logger.info("Indexing complete: %d chunks from %s", total, directory)
        return total

    def index_files(self, file_contents: dict[str, str]) -> int:
        """Index specific files (useful for indexing from GitHub API).

        Args:
            file_contents: Dict of {file_path: file_content}.

        Returns:
            Number of chunks indexed.
        """
        documents = []
        for path, content in file_contents.items():
            if not content.strip():
                continue
            chunks = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[{
                    "source": path,
                    "language": Path(path).suffix.lstrip("."),
                }],
            )
            documents.extend(chunks)

        if documents:
            self.vectorstore.add_documents(documents)
        return len(documents)

    def search(self, query: str, filename: str = "", k: int = 3) -> list[str]:
        """Search the vector store for relevant code context.

        Args:
            query: The diff or code snippet to find context for.
            filename: Current file being reviewed (boosts results from same/related files).
            k: Number of results to return.

        Returns:
            List of relevant code snippets with source info.
        """
        try:
            # Search with optional metadata filter for related files
            results = self.vectorstore.similarity_search(query, k=k)

            formatted = []
            for doc in results:
                source = doc.metadata.get("source", "unknown")
                formatted.append(f"# From: {source}\n{doc.page_content}")

            return formatted
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return []

    def clear(self) -> None:
        """Clear the vector store (for re-indexing)."""
        try:
            self.vectorstore.delete_collection()
            self._vectorstore = None
            logger.info("Vector store cleared")
        except Exception as e:
            logger.warning("Failed to clear vector store: %s", e)
