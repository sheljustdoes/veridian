# veridian.

**veridian.** is a corpus-grounded research cognition engine.

Its purpose is not to replace scholarship, automate expertise, or compress mastery into hours.

Its purpose is to reduce analysis paralysis and provide structured epistemic reflection when entering a research domain.

---

## Why veridian. Exists

Modern research fields are dense, fragmented, and rapidly expanding.

When entering a topic, researchers often face:

* Hundreds or thousands of papers
* Unclear conceptual hierarchies
* Competing interpretations
* No obvious starting point
* Hidden foundational assumptions

veridian. aims to:

* Map the intellectual terrain of a research topic
* Identify dominant, emerging, and minority interpretations
* Surface structural relationships between papers
* Provide reflective feedback on articulated understanding
* Encourage epistemic humility and proportional reasoning

It is not a learning shortcut.

It is an orientation and reflection system.

---

## Core Principles

veridian. is built on five principles:

1. **Corpus-Bounded Claims**
   All reflection is grounded in retrieved literature.
   The system never claims universal truth — only corpus-relative structure.

2. **Reflective, Not Evaluative**
   veridian. does not grade.
   It surfaces omissions, alignment, and unsupported claims.

3. **Intellectual Friction Is Healthy**
   Unsupported or weakly supported claims are flagged transparently.
   Confidence should track evidence density.

4. **Interpretive Landscape Awareness**
   Research fields contain dominant, emerging, and minority views.
   These are contextualized, not flattened.

5. **User Agency**
   The user remains the final arbiter.
   veridian. provides structure, not authority.

6. **Context Engineering Over Prompting**
   veridian structures retrieval, claims, entities, and graph relationships so downstream reasoning is grounded in explicit context.

---

## What veridian. Does (Current Prototype)

1. Accepts a research query.
2. Retrieves abstracts from PubMed.
3. Embeds and clusters the corpus.
4. Generates semantic summaries of cluster themes.
5. Visualizes the research terrain.
6. Accepts free-form explanation from the user.
7. Extracts atomic claims.
8. Maps claims to corpus clusters.
9. Extracts named entities (concepts, methods, authors, institutions) from claims.
10. Resolves entities into canonical graph nodes (entity resolution).
11. Builds a knowledge graph where cluster nodes are linked by semantic similarity and entity nodes connect to aligned clusters.
12. Reflects alignment and unsupported statements.

---

## Technical Architecture (Context Engineering)

veridian now uses explicit context-engineering primitives:

* **Ontology-oriented node model**
   The graph schema includes typed nodes: `cluster` and `entity` (with entity subtypes such as concept, method, author, institution).

* **Knowledge graph layer**
   Cluster embeddings are converted into explicit graph edges when centroid similarity exceeds a threshold.
   This makes latent vector relationships queryable as structured relational knowledge.

* **Entity extraction + entity resolution**
   Atomic claims are parsed for named entities, then normalized/deduplicated into canonical entities with aliases and mention counts.

* **Claim-to-graph grounding**
   Claims are first aligned to clusters; entities inherit that alignment and are linked to the relevant cluster nodes.

* **Context engineering loop**
   Retrieval → embeddings → clusters → claims → entities → graph.
   Each stage constrains the next stage with corpus-grounded context rather than unconstrained generation.

This is a structural prototype, not a finished product.

---

## Capability Translation: Semantic Modeling and Context Engineering

This project is intentionally framed as an **AI-first context architecture** system, not only an NLP demo.
It demonstrates how semantic modeling and context engineering can improve downstream model behavior, retrieval quality, and interpretability in high-dimensional scientific domains.

### Semantic architecture & AI-first context modeling

* **Enterprise-style semantic representation (prototype scope)**
   veridian defines typed domain entities (`cluster`, `entity`) and explicit relationships (semantic similarity, claim-grounded entity linkage) as a reusable semantic layer for AI consumption.

* **Domain-entity modeling posture (life-sciences aligned)**
   The same schema pattern is designed to extend to enterprise healthcare concepts such as payer, provider, patient, product, site, and indication, with explicit relationship contracts for AI reasoning.

* **Semantic schema / ontology patterns**
   The graph model captures entities, relationship types, and constraints (thresholded similarity, cluster-aligned entity links), forming a lightweight ontology foundation that can evolve toward formal RDF/OWL exports.

* **Context engineering standards (implemented pattern)**
   The pipeline enforces a consistent context path:
   retrieval → embedding → clustering → claim extraction → entity resolution → graph grounding → reflective output.
   This pattern is designed so models operate on explicit, structured context instead of unconstrained prompts.

* **Prompt/tool/memory/retrieval shaping**
   Outputs are structured for AI consumption as grounded claims, aligned entities, and graph relationships, establishing a basis for prompt assembly, tool orchestration, session memory, retrieval indices, and structured downstream outputs.

### Feature engineering & model performance orientation

* **Feature-centric design**
   Core features include embedding vectors, cluster centroids, claim-cluster similarity scores, entity mention counts, and graph edge weights.

* **Model reliability framing**
   Claim alignment thresholds function as an initial guardrail against unsupported statements, with clear extension points for leakage checks, stability monitoring, and explainability instrumentation.

### Context-aware ML / GenAI / RL-informed system thinking

* **Context-aware GenAI integration**
   LLM calls are constrained by corpus-derived structure (cluster and claim context), and outputs are normalized into typed artifacts (claims/entities) for downstream reasoning.

* **RL-informed roadmap alignment**
   The architecture supports future reward signals such as grounding coverage, semantic consistency, and interpretive diversity to guide ranking/orchestration decisions.

### Retrieval, knowledge, memory, and governance foundations

* **Retrieval + knowledge graph hybrid**
   veridian combines vector retrieval structure (embeddings/clusters) with explicit relational knowledge (graph edges), making both similarity and topology queryable.

* **Semantic quality gate trajectory**
   The current system surfaces grounding confidence and unsupported claims; next quality gates are entity completeness, relationship validity, and taxonomy drift detection.

### Cross-functional translation value

* **Roadmap-ready decomposition**
   The architecture cleanly separates retrieval, semantic modeling, reasoning context, and reflective outputs, making it straightforward to partner with engineering, product, and governance stakeholders on productionization.

* **Program-ready communication artifact**
   This README is structured to communicate semantic definitions, context standards, reliability tradeoffs, and implementation maturity in language used by enterprise AI programs.

---

## What veridian. Is Not

* A literature review generator
* A summary engine
* A citation counter
* A course replacement
* A grading system
* A shortcut to expertise

It is a scaffold for structured thinking.

---

## Long-Term Vision

veridian. may evolve to include:

* Citation-weighted cluster centrality
* Dominant / emerging / minority interpretation labeling
* Progressive exposure of debate structure
* Iterative articulation loops
* Transparent evidence density indicators
* Interactive research terrain exploration

The goal is not speed.

The goal is clarity.

---

## Design Ethos

veridian. assumes:

* Expertise is built through articulation and revision.
* Confidence should be proportional to evidence.
* Scientific discourse is landscape-based, not binary.
* Reflection strengthens scholarship.

---

## Current Status

Early prototype.

Core pipeline implemented:

* Retrieval
* Embedding
* Clustering
* Claim extraction
* Entity extraction + entity resolution
* Knowledge graph construction
* Alignment reflection
* Basic visualization

This repository represents the structural foundation.

---

## Web Experience (New)

The project now includes a webpage mode with the following flow:

1. Enter a research search term.
2. Add your reasoning/justification for the search.
3. See a loading state while veridian retrieves, embeds, and clusters documents.
4. View an embedded, interactive cluster map and grouped cluster summaries.

### Run the webpage

For a pseudo-static interactive demo (no API key, no backend), just open `index.html` in your browser.

Optional local static server:

```bash
cd /Users/shel/Github/veridian
python -m http.server 8000
```

Then open:

`http://127.0.0.1:8000`

### Run live backend mode (optional)

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python veridian.py --web --host 127.0.0.1 --port 8000
```

Then open:

`http://127.0.0.1:8000`

### Notes

* Static portfolio/demo mode does **not** require `OPENAI_API_KEY`.
* Live backend retrieval mode requires `OPENAI_API_KEY` in `.env` (or environment).
* Search results are constrained to 25-120 PubMed records per run (default 100).
* Retrieved abstracts are persisted to `corpus.json`.

---

## Deploy on GitHub Pages

GitHub Pages can only host static files. veridian needs a Python backend for PubMed + OpenAI calls.

Use a split deployment:

1. **Frontend** (`index.html`) on GitHub Pages.
2. **Backend** (`veridian.py`) on a Python host (Render, Railway, Fly.io, etc.).

### Quick deployment checklist

1. Create a GitHub repo for this folder and push your code.
2. Deploy backend with `OPENAI_API_KEY` configured.
3. Enable GitHub Pages deployment for the frontend.
4. Open your GitHub Pages URL with `apiBase=<your_backend_url>`.

### 1) Deploy backend (example start command)

```bash
python veridian.py --web --host 0.0.0.0 --port $PORT
```

Set environment variable on your host:

```bash
OPENAI_API_KEY=your_key_here
```

### 2) Publish frontend to GitHub Pages

This repository now includes an automated Pages workflow:

`/.github/workflows/deploy-pages.yml`

After you push to `main`, GitHub Actions deploys `index.html` automatically.

One-time GitHub setup:

* **Settings → Pages**
* **Build and deployment → Source: GitHub Actions**

If your default branch is not `main`, update `branches: [main]` in the workflow file.

### 3) Connect Pages frontend to backend API

Open your GitHub Pages URL with `apiBase` query param:

```text
https://<your-username>.github.io/<repo>/?apiBase=https://<your-backend-domain>
```

The frontend stores this value and uses it for `/api/search` requests.

### 4) First push commands (if needed)

```bash
cd /Users/shel/Github/veridian
git init
git add .
git commit -m "Initial veridian deploy setup"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo>.git
git push -u origin main
```

### Portfolio mode (neutered functionality)

For portfolio/public website use, the frontend now supports a built-in **portfolio mode**:

* No PubMed requests
* No OpenAI calls
* No backend dependency
* Simulated loading + demo cluster visualization

Behavior:

* Portfolio mode is enabled by default unless both `live=1` and `apiBase` are set.
* This keeps the site pseudo-static by default on localhost and GitHub Pages.

Useful URLs:

```text
https://<your-username>.github.io/<repo>/
```

(Auto portfolio mode)

```text
https://<your-username>.github.io/<repo>/?portfolio=1
```

(Force portfolio mode)

```text
https://<your-username>.github.io/<repo>/?apiBase=https://<your-backend-domain>&live=1
```

(Force live backend mode)
