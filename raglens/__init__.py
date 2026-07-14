"""RagLens — dashboard de qualidade de RAG.

Nucleo GENERICO (nao sabe nada de nenhum RAG especifico). Um RAG vira mensuravel
implementando o contrato em `raglens.adapter.RagAdapter`. O adapter do Second Brain
(primeiro cliente/dogfood) vive em `adapters/sb_adapter.py`, fora do pacote generico.

Duas familias de metrica:
- Saude do indice (Fase 1): cobertura, staleness, orfaos. Zero chamada de API.
- Qualidade de retrieval (Fase 3): hit@k, MRR, recall@k contra um gabarito.
"""

__version__ = "0.1.0"
