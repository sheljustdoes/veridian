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
9. Reflects alignment and unsupported statements.

This is a structural prototype, not a finished product.

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
* Alignment reflection
* Basic visualization

This repository represents the structural foundation.

---

## Web Experience (New)

The project now includes a webpage mode with the following flow:

1. Enter a research search term.
2. Add your reasoning/justification for the search.
3. See a loading state while Veridian retrieves, embeds, and clusters documents.
4. View an embedded, interactive cluster map and grouped cluster summaries.

### Run the webpage

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

* Web mode still requires `OPENAI_API_KEY` in `.env` (or environment).
* Search results are constrained to 25-50 PubMed records per run.
* Retrieved abstracts are persisted to `corpus.json`.

---

## Deploy on GitHub Pages

GitHub Pages can only host static files. Veridian needs a Python backend for PubMed + OpenAI calls.

Use a split deployment:

1. **Frontend** (`index.html`) on GitHub Pages.
2. **Backend** (`veridian.py`) on a Python host (Render, Railway, Fly.io, etc.).

### 1) Deploy backend (example start command)

```bash
python veridian.py --web --host 0.0.0.0 --port $PORT
```

Set environment variable on your host:

```bash
OPENAI_API_KEY=your_key_here
```

### 2) Publish frontend to GitHub Pages

Push this repo to GitHub, then enable Pages:

* **Settings → Pages**
* **Build and deployment → Source: Deploy from a branch**
* Select branch (for example `main`) and folder (`/ (root)`)

### 3) Connect Pages frontend to backend API

Open your GitHub Pages URL with `apiBase` query param:

```text
https://<your-username>.github.io/<repo>/?apiBase=https://<your-backend-domain>
```

The frontend stores this value and uses it for `/api/search` requests.

### Portfolio mode (neutered functionality)

For portfolio/public website use, the frontend now supports a built-in **portfolio mode**:

* No PubMed requests
* No OpenAI calls
* No backend dependency
* Simulated loading + demo cluster visualization

Behavior:

* On `github.io`, portfolio mode is enabled automatically unless `live=1` is set.
* If no `apiBase` is configured, portfolio mode is also used.

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
