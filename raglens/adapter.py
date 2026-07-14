"""Contrato que qualquer RAG implementa pra ser medido pelo RagLens.

O nucleo generico so fala com o RAG por esta interface. Trocar Second Brain por
Pinecone/LlamaIndex/qualquer coisa = escrever um novo adapter; health.py e
metrics.py nao mudam uma linha. E o seam que separa "produto" de "meu scriptzinho".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDoc:
    """Verdade: um documento que DEVERIA estar indexado."""

    doc_id: str
    mtime: float  # ultima modificacao do arquivo-fonte (epoch)


@dataclass(frozen=True)
class IndexedDoc:
    """O que ESTA indexado, com o snapshot registrado no momento da indexacao."""

    doc_id: str
    indexed_mtime: float  # mtime do fonte no instante em que foi indexado
    n_chunks: int
    tokens: int = 0


class RagAdapter(ABC):
    """Ponte entre o RagLens e um RAG concreto.

    - `source_docs` + `indexed_docs` bastam pra saude do indice (Fase 1) — sem API.
    - `search` so e exigido pra eval de qualidade (Fase 3).
    """

    name: str = "rag"

    @abstractmethod
    def source_docs(self) -> list[SourceDoc]:
        """Conjunto-verdade: tudo que deveria estar no indice."""

    @abstractmethod
    def indexed_docs(self) -> list[IndexedDoc]:
        """O que o indice realmente contem."""

    def search(self, query: str, k: int = 5) -> list[str]:
        """Top-k doc_ids pra uma query. Opcional — so pra Fase 3."""
        raise NotImplementedError(f"adapter '{self.name}' nao implementa search()")
