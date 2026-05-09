# PaperBridge: Sistem Mimari & Kullanım Rehberi

> **Tarih:** 2026-05-01
> **Durum:** Faz 1 + Faz 2 tamamlandi, gercek veri ile test edildi
> **Korpus:** 9,987 PDF → 175,349 chunk → 3,687 konu

---

## Bu Sistem Ne Yapiyor?

**192 akademik PDF** (3D bin packing / container loading / optimization alani) uzerinde iki temel islev:

| Modul | Ne Yapiyor | Input | Output |
|-------|-----------|-------|--------|
| **Faz 1: Akilli Sorgu** | PDF'lere soru sor → kanitlarla desteklenmis cevap al | "3D bin packing algoritmalar neler?" | Cevap metni + 5 kanit chunk |
| **Faz 2: Research Gap** | Korpusu analiz et → hangi konularda bosluk var | Tum 175K chunk | Gap raporu (JSON + Markdown) |

---

## Altyapı

| Servis | URL | Model | Ne Ise Yarar |
|--------|-----|-------|--------------|
| **LLM** | `192.168.0.28:8005` | Qwen3.6-35B-A3B | Soru-cevap, arama query uretimi, gap validasyonu |
| **Qdrant** | `192.168.0.28:6333` | — | 175,349 chunk, 4096-dim embedding (vektor veritabani) |
| **Embedding** | `localhost:8082` | Qwen3-Embedding-8B | Metin → 4096 boyutlu vektor |
| **Tantivy** | `./data/tantivy_index/` | — | 192 PDF tam metin arama |
| **OpenAlex** | `api.openalex.org` | — | Global akademik konu taksonomisi |

---

## GERCEK TEST SONUCLARI

### Sorgu Testi (Canli)

**INPUT:**
```
"What are the main algorithms and approaches for 3D bin packing and container loading problems?"
```

**OUTPUT:**
- **Sure:** 436 saniye (7.3 dakika)
- **Status:** SUCCESS
- **Kanit:** 5 chunk

**CEVAP (ozet):**
```
Exact methods: branch-and-bound, branch-and-cut, MIP, constraint programming
Heuristic: layer-building, block arrangement, greedy, caving degree
Metaheuristic: GA, Tabu Search, Simulated Annealing, GRASP, VNS, PSO
Hybrid: Column generation + heuristic, GRASP + VND
ML: Deep reinforcement learning (DRL), Physics simulation
```

**KANITLAR (ilk 3):**
```
[1] Score: 8 — "main algorithms include exact methods like branch-and-bound,
                metaheuristics including tabu search, genetic algorithms,
                differential evolution, large neighborhood search..."

[2] Score: 7 — "key methodologies include column generation-based heuristics,
                hybrid genetic algorithms, GRASP/VND hybrids..."

[3] Score: 7 — "frequent use of heuristic and metaheuristic approaches,
                including genetic algorithms, simulated annealing, tabu search..."
```

### Gap Analizi Testi (Canli)

```
Input:  175,349 chunk
Sure:   ~65 dakika (tum pipeline)
Output: 9 confirmed gap + 6 rejected
```

---

## Faz 1: Akıllı Sorgu (RAG Pipeline)

### Akıs Diyagrami

```
Kullanici Sorusu
      ↓
┌──────────────────────────────────────────────────────────┐
│  paperbridge_agent_query()                               │
│                                                          │
│  Adim 1: LLM → 3 arama query uretir                     │
│    Orn: "What are heuristics for 3D bin packing?"        │
│                                                          │
│  Adim 2: Qdrant → semantik arama (4096-dim)             │
│    15 en benzer chunk doner                              │
│                                                          │
│  Adim 3: Tantivy → keyword arama                         │
│    Ilgili PDF hash'leri toplar                           │
│                                                          │
│  Adim 4: Qdrant → ilgili PDF'lerin chunk'larini ceker   │
│    (max 10 chunk/PDF)                                    │
│                                                          │
│  Adim 5: agent_evidence() → chunk'lari puanla           │
│    Hangi chunk soruyla ilgili? (score 1-9)              │
│                                                          │
│  Adim 6: aquery() → LLM'e kanitlari gonder              │
│    "Bu chunk'lara gore cevabin nedir?"                   │
│                                                          │
│  Adim 7: AnswerResponse doner                            │
│    .session.answer   → cevap metni                      │
│    .session.contexts → kanit chunk'lari (list[Context]) │
│    .status          → SUCCESS / UNSURE                  │
└──────────────────────────────────────────────────────────┘
      ↓
  Cevap + Kanitlar
```

### Kod Ornegi

```python
import asyncio
from paperqa.stores import paperbridge_agent_query

async def main():
    result = await paperbridge_agent_query(
        query="What are the main algorithms for 3D bin packing?",
        llm_api_base="http://192.168.0.28:8005/v1",
    )

    print("CEVAP:", result.session.answer)
    print("Status:", result.status)
    print("Kanit sayisi:", len(result.session.contexts))

    for ctx in result.session.contexts:
        print(f"  Score: {ctx.score} | ID: {ctx.id}")
        print(f"  Ozet:  {ctx.context[:200]}...")

asyncio.run(main())
```

### Çalıştırma

```bash
cd paper-qa
export OPENAI_API_KEY="sk-local-dummy"
PYTHONPATH=src python3 -c '
import asyncio
from paperqa.stores import paperbridge_agent_query

async def m():
    r = await paperbridge_agent_query(
        query="3D bin packing algoritmalar?",
        llm_api_base="http://192.168.0.28:8005/v1"
    )
    print(r.session.answer)

asyncio.run(m())
'
```

### Celişki Tespiti (ContraCrow)

```python
from paperqa.stores import paperbridge_contracrow

result = await paperbridge_contracrow(
    claim="Genetik algoritmalar 3D bin packing'ta en iyi sonuc verir",
    llm_api_base="http://192.168.0.28:8005/v1",
)
# Donus: SUPPORTED / CONTRADICTED / INSUFFICIENT_EVIDENCE
```

---

## Faz 2: Research Gap Analizi

### Akıs Diyagrami

```
Korpus (175K chunk)
      ↓
┌───────────────────────────────────────────────────────────┐
│  run_gap_analysis() — 5 component                        │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ 1. BERTopic — Konu Kesfi                            │ │
│  │   Input:  175,349 chunk × 4096-dim                  │ │
│  │   Sure:   ~5 dakika                                 │ │
│  │                                                   │ │
│  │   Adimlar:                                         │ │
│  │     PCA(4096→256) → UMAP(256→10) → HDBSCAN        │ │
│  │                                                   │ │
│  │   Output:                                          │ │
│  │     3,687 konu                                     │ │
│  │     27,160 outlier (konusuz)                       │ │
│  │     2,872 sparse (<50 chunk) → gap adidi           │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ 2. Gap-Language — Yazar Ipuculari                  │ │
│  │   Input:  Tantivy index                             │ │
│  │   Sure:   ~1 dakika                                 │ │
│  │                                                   │ │
│  │   Arama: "future work", "limitation",              │ │
│  │          "not been studied", "remains unclear"     │ │
│  │                                                   │ │
│  │   Output:                                          │ │
│  │     286 chunk bulundu                              │ │
│  │     8 kategori (BERTopic ile)                      │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ 3. OpenAlex — Global Karşilastirma                 │ │
│  │   Input:  OpenAlex API                             │ │
│  │   Sure:   ~37 saniye                               │ │
│  │                                                   │ │
│  │   Output:                                          │ │
│  │     729 global konu cekildi                        │ │
│  │     31 domain-relevant konu                        │ │
│  │     164 konu korpus'ta eksik                       │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ 4. Novelpy — Yeni Kombinasyonlar                   │ │
│  │   Input:  BERTopic konseptleri                     │ │
│  │   Sure:   ~10 saniye                               │ │
│  │                                                   │ │
│  │   Adimlar:                                         │ │
│  │     Konsept cikar → Co-occurrence matrix →        │ │
│  │     Atypical pair'leri bul                         │ │
│  │                                                   │ │
│  │   Output:                                          │ │
│  │     10,115 konsept                                 │ │
│  │     50 yeni konsept cifti                          │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ 5. LLM Synthesis — Validasyon                      │ │
│  │   Input:  2,920 kandidat → 252 domain-relevant    │ │
│  │   Sure:   ~4-5 dk/kandidat                        │ │
│  │                                                   │ │
│  │   Her biri icin:                                   │ │
│  │     paperbridge_agent_query("Bu konuda ne var?") │ │
│  │     "Bilmiyorum" → CONFIRMED GAP                  │ │
│  │     "Iste kanitlar" → REJECTED                    │ │
│  │                                                   │ │
│  │   Output:                                          │ │
│  │     9 confirmed gap + 6 rejected (15 test)       │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  Cikti: GapReport                                       │
│    .gaps[]         → her gap'in title/desc/confidence   │
│    .summary        → executive summary                  │
│    .recommendations  → oneriler                         │
│    JSON + Markdown dosyalari                            │
└───────────────────────────────────────────────────────────┘
```

### BERTopic Sonuclari (Gercek Veri)

```
Toplam konu:    3,687
Outlier:        27,160 chunk (hicbir konuya uymuyor)
Sparse:         2,872 konu (<50 chunk → gap adidi)
Medium:          811 konu
Large:             4 konu
PCA variance:   0.712 (4096→256)
```

**Top 20 Konu:**
```
 #0   461 doc   fuel emissions vehicle vrp routing        ← Vehicle Routing
 #1   242 doc   selfweight topology stress compliance     ← Topology Optimization
 #2   230 doc   pso particle gbest swarm velocity         ← PSO
 #3   210 doc   neighborhood tabu vnd dvnd                ← Tabu Search
 #4   198 doc   pso fasi swarm vpgd jaya                  ← PSO variants
 #5   189 doc   boxes multidrop clp bortfeldt             ← CLP! (Hedefimiz)
 #6   187 doc   dg dgs kw dstatcoms                       ← Power Systems
 #7   173 doc   controller lfc pid frequency              ← Control Systems
 #8   172 doc   lpsp coe sizing renewable                 ← Energy
 #9   171 doc   stock cutting leftovers leftover           ← Cutting Stock!
#10   168 doc   kubernetes pod pods scheduler             ← Kubernetes
#11   160 doc   collection waste weee garbage             ← Waste Collection
#12   151 doc   ttd bjiomega ils axle                     ← ?
#13   149 doc   ebusiness amazon business companies       ← E-business
#14   149 doc   0000 pareto igd nsgaii                    ← Multi-objective
#15   138 doc   facial face avatar hair                   ← Computer Vision
#16   137 doc   migrations recourse migration bins        ← Cloud Migration
#17   137 doc   agvs agv automated ushaped                ← AGV
#18   136 doc   inequalities rmp pricing nonrobust        ← Math Programming
#19   136 doc   mpi cfd stencil openacc                   ← HPC
```

### Gap-Language Sonuclari (Gercek Veri)

```
Toplam gap chunk:  286
Kategori sayisi:     8
```

| Kategori | Chunk | Anahtar Sozcukler | Domain Relevant? |
|----------|-------|-------------------|------------------|
| 0_3d_underexplored_for_is | 49 | 3d, underexplored, deep learning | PARTIAL |
| 1_packing_problem_open_bin | 45 | packing, problem, open, bin | **EVET** |
| 2_energy_for_and_of | 36 | energy, power, demand, management | HAYIR |
| 3_in_potential_and_cell | 30 | potential, cell, unclear, limitation | HAYIR |
| 4_routing_vehicle_problem_open | 21 | routing, vehicle, algorithm, path | **EVET** |
| 5_gpus_potential_and_for | 20 | gpus, work, future, limitation | HAYIR |
| 6_learning_reinforcement_to_yet | 11 | learning, reinforcement, mutation | **EVET** |
| 7_investigation_of_needs | 11 | investigation, numerical, ultrasonic | HAYIR |

### OpenAlex Sonuclari

```
Global konu:       729
Domain relevant:    31
Korpus'ta eksik:  164
```

**Domain-relevant global konular (top 8):**
```
Manufacturing Process Optimization:    246,312 work
Parallel Computing Optimization:       200,012 work
Transportation Planning Optimization:  115,361 work
Scheduling & Optimization Algorithms:  102,549 work
Logistics Optimization:                 98,496 work
Maritime Ports & Logistics:             85,429 work
Metaheuristic Optimization:             72,145 work
Freight Transport Logistics:            62,111 work
```

### Gap Validasyonu (LLM — Gercek Sonuclar)

**15 kandidat test edildi. Toplam sure: 62.3 dakika**

**CONFIRMED GAP (9):**
```
  #1 [0.60] 3D underexplored areas (49 chunk)
        LLM: "I cannot answer." → Korpus'ta YOK

  #2 [0.60] Energy future work (36 chunk)
        LLM: "Sadece bibliografik referanslar" → Icerik YOK

  #3 [0.60] Cell potential (30 chunk)
        LLM: "Sadece bibliografik referanslar" → Icerik YOK

  #4 [0.60] GPUs potential (20 chunk)
        LLM: "Sadece bibliografik referanslar" → Icerik YOK

  #5 [0.60] Investigation needs direction (11 chunk)
        LLM: "Sadece bibliografik referanslar" → Icerik YOK

  #6 [0.60] Boxes + pallet + superboxes (46 doc)
        LLM: "I cannot answer." → Konu cok az calisilmis

  #7 [0.60] Subcontainer + sparsest (46 doc)
        LLM: "I cannot answer." → Konu cok az calisilmis

  #8 [0.60] Novel: 00 + sg2 (atypicality=0.97)
        LLM: "I cannot answer." → Gürültü (hex kod)

  #9 [0.60] Novel: 00 + trimming (atypicality=0.97)
        LLM: "I cannot answer." → Gürültü (hex kod)
```

**REJECTED — KORPUSTA VAR (6):**
```
  #1 Open bin packing problem (ODBPP)
        LLM: "3D packing variant where container dimensions are
              not pre-specified but must be simultaneously optimized..."

  #2 Vehicle routing + 3D container loading
        LLM: "Integration of vehicle routing decisions with 3D
              container loading constraints constitutes a well-documented
              research domain..."

  #3 Reinforcement learning + bin packing
        LLM: "RL and DRL have been extensively applied to 3D bin
              packing, specifically targeting online decisions..."

  #4 Pallets + LTL truck cargo
        LLM: "LTL logistics involves consolidating partial shipments...
              Recent scholarship addresses this intersection..."

  #5 Pallet + PLP + boxes
        LLM: "Pallet Loading Problem (PLP) optimizes geometric
              arrangement of boxes onto pallets..."

  #6 Air cargo palletizing layouts
        LLM: "Research addresses air cargo palletizing layouts within
              3D bin packing, focusing on constraint integration..."
```

### Calistirma

```bash
# CLI ile tum pipeline
cd paper-qa
PYTHONPATH=src python3 scripts/run_gap_analysis.py \
    --output-dir ./data/gap_analysis/ \
    --llm-api-base http://192.168.0.28:8005/v1

# Python'dan
PYTHONPATH=src python3 -c '
import asyncio
from paperqa.stores import run_gap_analysis

async def m():
    report = await run_gap_analysis(
        qdrant_collection="paperbridge_glm_v2",
        output_dir="./data/gap_analysis/",
        llm_api_base="http://192.168.0.28:8005/v1",
    )
    for gap in report.gaps:
        print(f"[{gap.confidence:.2f}] {gap.title}")

asyncio.run(m())
'
```

---

## Dosya Yapisi

```
paper-qa/
├── src/paperqa/
│   └── stores/
│       ├── __init__.py              # Public exports
│       ├── paperbridge_store.py     # Faz 1: Qdrant adapter + Docs
│       ├── agent.py                 # Faz 1: Sorgu pipeline + ContraCrow
│       ├── gap_analysis.py          # Faz 2: Orchestrator + dataclass'lar
│       ├── bertopic_pipeline.py     # Faz 2: BERTopic konu kesfi
│       ├── gap_language.py          # Faz 2: Gap keyword tespiti
│       ├── openalex_compare.py      # Faz 2: OpenAlex karsilastirma
│       ├── novelpy_pipeline.py      # Faz 2: Novelty scoring
│       └── gap_synthesis.py         # Faz 2: LLM validasyon + rapor
│
├── scripts/
│   ├── build_tantivy_index.py       # Tantivy index olustur
│   └── run_gap_analysis.py          # Gap analysis calistir
│
├── tests/
│   ├── test_paperbridge_store.py         # Faz 1 testleri (22 test)
│   └── test_gap_analysis_integration.py  # Faz 2 testleri (10 test)
│
├── data/
│   ├── tantivy_index/paperbridge_index/paperbridge_index/index/
│   │   ├── meta.json                        # Tantivy meta
│   │   ├── index/                           # Inverted index
│   │   └── docs/                            # 192 dokuman
│   │
│   └── gap_analysis/                          # Ciktilar
│       ├── bertopic/
│       │   ├── bertopic_model/               # BERTopic model dosyalari
│       │   ├── metadata.json                 # 3,687 konu, 27,160 outlier
│       │   └── topic_info.csv                # 3,689 satir
│       ├── gap_language/
│       │   ├── categories.json               # 8 kategori
│       │   └── gap_chunks.json               # 286 chunk
│       ├── openalex/
│       │   ├── global_topics.json            # 729 konu
│       │   ├── coverage.json                 # Coverage analizi
│       │   └── missing_topics.json           # 164 eksik konu
│       ├── novelpy/
│       │   ├── concepts.json                 # 10,115 konsept
│       │   ├── cooccurrence.json             # Co-occurrence matrix
│       │   └── novel_pairs.json              # 50 yeni cift
│       ├── candidates.json                   # 2,920 toplam kandidat
│       ├── candidates_filtered.json          # 252 domain-relevant
│       ├── bertopic_summary.json             # BERTopic ozet
│       └── report/
│           ├── report.json                   # Gap raporu (JSON)
│           └── report.md                     # Gap raporu (Markdown)
│
└── docs/
    └── PAPERQA-ARCHITECTURE.md       # Bu dosya
```

---

## Veri Akisi (A-Z)

```
1. KAYNAK: 192 akademik PDF
              ↓  [PaperBridge Pipeline — once yapildi]
              ↓  Qwen3.5 VLM ile PDF → metin cikartma
              ↓  Her sayfa → 200 DPI görüntü → VLM analiz → section-tagged text

2. INDEKSLEME:
              ↓  Qdrant: 175,349 chunk × 4096-dim embedding
              ↓  Tantivy: 192 PDF full-text index

3. FAZ 1 — SORGU:
   Kullanici sorusu → LLM 3 query uretir
        → Qdrant semantik arama (15 chunk)
        → Tantivy keyword arama (PDF hash)
        → Qdrant ilgili PDF chunk'lari ceker (max 10/PDF)
        → Evidence scoring (hangi chunk ilgili?)
        → LLM aquery() → cevap uretir
        → Cevap + kanitlar doner (436 saniye ort.)

4. FAZ 2 — GAP ANALIZI:
   175K chunk → BERTopic (5dk) → 3,687 konu
        → Gap-Language (1dk)  → 286 gap chunk
        → OpenAlex (37sn)     → 729 global konu
        → Novelpy (10sn)      → 50 novel pair
        → Tum kandidatlari topla (2,920)
        → Filterle (252 domain-relevant)
        → LLM validasyon (4-5dk/each)
        → GapReport → JSON + Markdown

5. GELECEK — API + WEB UI:
        → FastAPI REST endpoints
        → Dashboard + arama + gap listesi
```

---

## Mevcut Problemler & Iyileştirmeler

| Problem | Neden | Cozum |
|---------|-------|-------|
| Novelpy gurusu | Hex kodlari/sayilar konsept olarak cikıyor (00, 000, 14dcbeq) | Concept extraction'da stopword + regex filter ekle |
| Gap isimleri | BERTopic otomatik isim (`0_3d_underexplored_for_is`) | Manuel isimlendirme veya LLM ile yeniden isimlendir |
| Tum gap'ler ayni confidence (0.6) | Tek kaynakli gap'ler ayni formül ile puanlanıyor | Multi-source cross-validation ekle |
| Sorgu sure: 7-8 dk | Her sorgu tam agent pipeline calistiriyor | Cache + fast mode ekle |
| BERTopic 3,687 konu cok fazla | min_cluster_size=3 cok kucuk | min_cluster_size=10-20 yap |
| OpenAlex 729→31 relevant | Onceden filtreleme yok | API sorgusunda domain keyword filter ekle |

---

## API Layer Tasarimi (Planlanan)

```python
# FastAPI endpoints

# Faz 1: Sorgu
POST /api/query
  Body:    {"query": "3D bin packing heuristics?", "timeout": 600}
  Return:  {"answer": "...", "contexts": [...], "status": "SUCCESS", "elapsed_s": 436}

GET  /api/query/{id}/status
GET  /api/query/{id}/result

# Faz 2: Gap Analizi
POST /api/gap-analysis/run
  Body:    {"max_candidates": 20, "sources": ["gap_language", "bertopic_sparse"]}
  Return:  {"job_id": "abc123", "status": "running"}

GET  /api/gap-analysis/status/{job_id}
GET  /api/gap-analysis/report/{job_id}
  Return: {"gaps": [...], "summary": "...", "metadata": {...}}

# Topluluk
GET  /api/topics
  Return: {"topics": [{"id": 5, "name": "boxes_multidrop_clp", "count": 189}]}

GET  /api/topics/{id}
  Return: {"id": 5, "words": ["boxes", "multidrop", "clp"], "papers": [...]}

GET  /api/topics/{id}/papers
  Return: {"papers": [{"title": "...", "year": 2023, "abstract": "..."}]}

# Gaps
GET  /api/gaps
  Return: {"gaps": [{"id": 1, "title": "...", "confidence": 0.6, "source": "gap_language"}]}

GET  /api/gaps/{id}
  Return: {"id": 1, "title": "...", "description": "...", "evidence": [...]}
```

---

## Test

```bash
cd paper-qa

# Tum testler
PYTHONPATH=src pytest tests/ -v -o addopts=""

# Sadece Faz 1
PYTHONPATH=src pytest tests/test_paperbridge_store.py -v -o addopts=""

# Sadece Faz 2
PYTHONPATH=src pytest tests/test_gap_analysis_integration.py -v -o addopts=""

# Network gerektirmeyen testler
PYTHONPATH=src pytest \
  tests/test_gap_analysis_integration.py::TestGapAnalysisDataclasses \
  tests/test_gap_analysis_integration.py::TestCandidateCollection \
  tests/test_gap_analysis_integration.py::TestNovelpyPipeline \
  tests/test_gap_analysis_integration.py::TestComputeCoverage \
  -v -o addopts=""
```

---

## Ortam Degiskenleri

```bash
# Zorunlu
export OPENAI_API_KEY="sk-local-dummy"       # Local LLM icin dummy key
export QDRANT_URL="http://192.168.0.28:6333"
export EMBEDDING_API_BASE="http://localhost:8082"
export LLM_URL="http://192.168.0.28:8005/v1"

# Opsiyonel
export QWEN35_VLLM_URL="http://192.168.0.28:8001/v1"
export QWEN35_MODEL_NAME="qwen/qwen3.5-9b"
export QWEN35_PAGE_DPI="200"
export ENABLE_RATE_LIMIT="false"
```

---

## Performans Referanslari

| Islem | Sure | Not |
|-------|------|-----|
| Tek sorgu (RAG) | 7-8 dakika | 5 kanit ile |
| BERTopic (tum korpus) | ~5 dakika | 175K chunk |
| Gap-Language extraction | ~1 dakika | 286 chunk |
| OpenAlex fetch | ~37 saniye | 729 konu |
| Novelpy | ~10 saniye | 50 novel pair |
| LLM gap validasyon | 4-5 dk/kandidat | 15 kandidat = 62 dk |
| Tum gap pipeline | ~1-2 saat | 252 kandidat ile |
