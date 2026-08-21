# RagLens

Dashboard de qualidade de RAG. Pluga o teu retrieval, vê se ele acha a coisa certa e afina por número — não por achismo.

**[→ Demo ao vivo](https://rodrigo-malavasi.github.io/raglens/)** — snapshot estático com dados reais. Os sliders funcionam: recalculam hit@k/MRR de verdade no browser, sem servidor.

O núcleo é **genérico**: não sabe nada de nenhum RAG específico. Um RAG vira mensurável implementando o contrato em `raglens/adapter.py`. O primeiro cliente (dogfood) é o Second Brain, via `adapters/sb_adapter.py`.

## Duas famílias de métrica

| Família | O que mede | Custo |
|---|---|---|
| **Saúde do índice** (Fase 1) | cobertura, staleness, órfãos | zero API — só lê o índice |
| **Qualidade de retrieval** (Fase 3) | hit@1, hit@5, MRR, recall@k contra um gabarito | 1 embed de query por pergunta |

A primeira responde "o índice está cobrindo o corpus?"; a segunda, "quando pergunto, ele acha?". As duas juntas separam *não achou porque não indexei* de *indexei e mesmo assim não achou* — que é a diferença entre um bug de pipeline e um problema de ranking.

## O ranker

Híbrido: cosseno sobre `voyage-4-lite` (1024 dims, sqlite-vec) + BM25 (FTS5) + um boost estrutural, combinados por peso. Os três componentes crus são devolvidos por `search_scored()` sem serem colapsados, o que permite **re-rankear sem re-embedar**: a query é embedada uma vez, e mexer nos pesos recombina os componentes já em memória. É o que faz o tuning ao vivo custar zero API e funcionar até na página estática.

## Como ler os números da demo

O corpus da demo são 104 documentos de arquitetura (~396k tokens, 1330 chunks) e o gabarito tem 55 perguntas.

| | |
|---|---|
| cobertura / freshness | 100% / 100% |
| hit@1 | 0.855 |
| hit@5 | 1.000 |
| MRR | 0.924 |
| recall@5 | 0.982 |

**Esses números são otimistas por construção, e isso é intencional dizer.** O gabarito foi escrito sobre o mesmo corpus que ele avalia, então mede *ranking*, não generalização — um golden set de produção precisa vir de queries reais de usuário, não do autor do corpus. O que o número prova é que a régua discrimina, e a aba **Ajuste** mostra isso: zerando o léxico (`w=1/0/0`) o hit@1 cai pra 0.782; zerando o vetorial (`w=0/1/0`) desaba pra 0.455 e o MRR pra 0.599. Uma métrica que não distinguisse ranker bom de ruim daria o mesmo valor nos três casos.

## Uso

```bash
python -m raglens health              # saúde do índice (texto)
python -m raglens health --json       # mesma coisa em JSON (pra tela/CI)
python -m raglens eval                # hit@k/MRR contra o gabarito
python -m raglens serve               # dashboard ao vivo em :7878
```

`--adapter sb` é o default. `SB_SCRIPTS` sobrescreve onde procurar `sb_vector_lib` (default `~/.claude/scripts`); `SB_ROOT` sobrescreve a raiz do vault.

### Gerar a versão publicável

```bash
python tools/build_static.py --print-paths      # audita: tudo que sairia no HTML
python tools/build_static.py --out docs/index.html
```

`serve.py` é um `ThreadingHTTPServer` de stdlib com rotas de escrita e sem auth — feito pra `localhost`, não pra internet. A build estática existe pra publicar sem ele: os números vão embutidos no HTML, não há servidor, e `--print-paths` lista exatamente o que ficaria visível antes de qualquer commit. O que sai é path e número; texto de documento, nunca.

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
│   ├── eval.py         harness: roda o gabarito contra um ranker
│   ├── report.py       formatação humana
│   ├── serve.py        dashboard ao vivo (Fase 4, localhost)
│   ├── web/            a tela — roda ao vivo ou como snapshot estático
│   └── cli.py          python -m raglens
├── adapters/
│   └── sb_adapter.py   implementação do Second Brain
├── tools/
│   └── build_static.py gera o HTML autocontido publicável
└── tests/
```

## Roadmap

- [x] Fase 1 — saúde do índice (cobertura/staleness/órfãos)
- [x] Fase 2 — golden set (gabarito de perguntas, gerado + curado)
- [x] Fase 3 — harness de qualidade (hit@k/MRR/recall rodando no gabarito)
- [x] Fase 4 — tela (gráfico, gerar pergunta, mexer peso ao vivo)
- [ ] Fase 5 — segundo adapter (provar que o seam segura um RAG que não é o do autor)
- [ ] Fase 6 — golden set de queries reais, separado do corpus
