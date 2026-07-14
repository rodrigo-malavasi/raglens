# RagLens

Dashboard de qualidade de RAG. Pluga o teu retrieval, vê se ele acha a coisa certa e afina por número — não por achismo.

O núcleo é **genérico**: não sabe nada de nenhum RAG específico. Um RAG vira mensurável implementando o contrato em `raglens/adapter.py`. O primeiro cliente (dogfood) é o Second Brain, via `adapters/sb_adapter.py`.

## Duas famílias de métrica

| Família | O que mede | Custo |
|---|---|---|
| **Saúde do índice** (Fase 1) | cobertura, staleness, órfãos | zero API — só lê o índice |
| **Qualidade de retrieval** (Fase 3) | hit@1, hit@5, MRR, recall@k contra um gabarito | 1 embed de query por pergunta |

## Uso

```bash
# da raiz do repo
python -m raglens health              # saúde do índice do SB (texto)
python -m raglens health --json       # mesma coisa em JSON (pra tela/CI)
```

`--adapter sb` é o default. `SB_SCRIPTS` sobrescreve onde procurar `sb_vector_lib` (default `~/.claude/scripts`); `SB_ROOT` sobrescreve a raiz do vault.

## Testes

```bash
python tests/test_metrics.py
```

## Estrutura

```
raglens/
├── raglens/            núcleo genérico (o produto)
│   ├── adapter.py      contrato RagAdapter — o seam
│   ├── health.py       saúde do índice (Fase 1)
│   ├── metrics.py      hit@k / MRR / recall@k (Fase 3, funções puras)
│   ├── report.py       formatação humana
│   └── cli.py          python -m raglens
├── adapters/
│   └── sb_adapter.py   implementação do Second Brain
└── tests/
```

## Roadmap

- [x] Fase 1 — saúde do índice (cobertura/staleness/órfãos)
- [ ] Fase 2 — golden set (gabarito de perguntas, gerado + curado)
- [ ] Fase 3 — harness de qualidade (hit@k/MRR/recall rodando no gabarito)
- [ ] Fase 4 — tela (HTML + servidor Python: gráfico, gerar pergunta, mexer peso ao vivo)

Doc do produto no Second Brain: `ORGANIZATIONS/FOURCORE/SISTEMAS/RAGLENS/`.
