import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import umap
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import requests
import xml.etree.ElementTree as ET
import time
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


# -----------------------------
# CONFIG
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
N_CLUSTERS = 4
SIMILARITY_THRESHOLD = 0.35  # tune this experimentally
CLUSTER_EDGE_SIMILARITY_THRESHOLD = 0.55
MIN_RESULTS = 25
MAX_RESULTS = 120
DEFAULT_RESULTS = 100


def load_environment():
    for env_filename in (".env", ".ENV"):
        env_path = Path(env_filename)
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"Loaded environment from {env_filename}")
            return

    load_dotenv()


load_environment()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def require_openai_client():
    if client is None:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Live retrieval mode requires OpenAI. "
            "Use portfolio/static mode in the frontend, or set OPENAI_API_KEY in .env/.ENV."
        )

# -----------------------------
# LOAD CORPUS
# -----------------------------
def load_corpus(path="corpus.json"):
    with open(path, "r") as f:
        return json.load(f)

def search_pubmed(query, max_results=200):
    print(f"Searching PubMed for: {query}")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }

    response = requests.get(base_url, params=params)
    data = response.json()
    pmids = data["esearchresult"]["idlist"]

    print(f"Found {len(pmids)} PMIDs")
    return pmids


def fetch_abstracts(pmids):
    print("Fetching abstracts...")
    abstracts = []
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    for i in range(0, len(pmids), 100):
        batch = pmids[i:i+100]

        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml"
        }

        response = requests.get(base_url, params=params)
        root = ET.fromstring(response.content)

        for article in root.findall(".//PubmedArticle"):
            try:
                title = article.find(".//ArticleTitle").text
                abstract_text = " ".join(
                    [ab.text for ab in article.findall(".//AbstractText") if ab.text]
                )
                year_elem = article.find(".//PubDate/Year")
                year = int(year_elem.text) if year_elem is not None else None

                if abstract_text:
                    abstracts.append({
                        "title": title,
                        "abstract": abstract_text,
                        "year": year
                    })
            except Exception:
                continue

        time.sleep(0.34)  # NCBI rate limit safety (~3 requests/sec)

    print(f"Retrieved {len(abstracts)} abstracts with text")
    return abstracts

# -----------------------------
# EMBEDDING
# -----------------------------
def embed_texts(texts):
    embeddings = []
    for text in tqdm(texts):
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=text
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)

# -----------------------------
# CLUSTERING
# -----------------------------
def cluster_embeddings(embeddings, n_clusters=N_CLUSTERS):
    if embeddings.size == 0:
        raise ValueError("No embeddings available for clustering.")

    n_samples = len(embeddings)
    effective_clusters = min(n_clusters, n_samples)

    if effective_clusters < 1:
        raise ValueError("Need at least one embedding to perform clustering.")

    if effective_clusters < n_clusters:
        print(
            f"Requested {n_clusters} clusters but only {n_samples} samples available. "
            f"Using {effective_clusters} cluster(s)."
        )

    kmeans = KMeans(n_clusters=effective_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    return labels, centroids


def summarize_cluster(cluster_id, corpus, labels, rationale=""):
    cluster_texts = [
        corpus[i]["abstract"]
        for i in range(len(labels))
        if labels[i] == cluster_id
    ]

    if not cluster_texts:
        return "Title: Empty Cluster\nSummary: No abstracts were assigned to this cluster."

    sample_text = "\n\n".join(cluster_texts[:5])

    rationale_line = f"User search rationale: {rationale}\n" if rationale else ""

    prompt = f"""
Below are abstracts from a research cluster.
{rationale_line}

Summarize the shared theme in 2-3 sentences.
Provide:
1. A short descriptive cluster title.
2. A concise summary paragraph.

Abstracts:
{sample_text}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content


def visualize_clusters(embeddings, labels):
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        cmap="tab10"
    )

    plt.title("Veridian Cluster Map")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.show()


def compute_umap_points(embeddings):
    reducer = umap.UMAP(n_components=2, random_state=42)
    return reducer.fit_transform(embeddings)

# -----------------------------
# CLAIM EXTRACTION
# -----------------------------
import re

def extract_claims(explanation):
    prompt = f"""
Extract atomic declarative claims from the following explanation.
Return ONLY valid JSON. No markdown. No commentary.
Return a JSON array of short declarative statements.

Explanation:
{explanation}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    claims_text = response.choices[0].message.content

    # Extract JSON array using regex
    match = re.search(r"\[.*\]", claims_text, re.DOTALL)
    if match:
        claims_text = match.group(0)

    try:
        claims = json.loads(claims_text)
    except Exception:
        print("⚠️ Could not parse claims cleanly. Raw output:")
        print(claims_text)
        claims = []

    return claims


def extract_entities_from_claims(claims):
    if not claims:
        return []

    numbered_claims = "\n".join([f"{idx + 1}. {claim}" for idx, claim in enumerate(claims)])
    prompt = f"""
Extract named entities from the claims below.
Entity types must be one of: concept, method, author, institution, other.
Return ONLY valid JSON. No markdown. No commentary.

Return a JSON array where each item has:
- name (string)
- type (string)
- source_claim (exact claim text)

Claims:
{numbered_claims}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    entities_text = response.choices[0].message.content
    match = re.search(r"\[.*\]", entities_text, re.DOTALL)
    if match:
        entities_text = match.group(0)

    try:
        raw_entities = json.loads(entities_text)
    except Exception:
        print("⚠️ Could not parse entities cleanly. Raw output:")
        print(entities_text)
        return []

    cleaned_entities = []
    valid_types = {"concept", "method", "author", "institution", "other"}
    for entity in raw_entities:
        if not isinstance(entity, dict):
            continue

        name = (entity.get("name") or "").strip()
        entity_type = (entity.get("type") or "other").strip().lower()
        source_claim = (entity.get("source_claim") or "").strip()

        if not name:
            continue

        if entity_type not in valid_types:
            entity_type = "other"

        cleaned_entities.append({
            "name": name,
            "type": entity_type,
            "source_claim": source_claim
        })

    return cleaned_entities


def normalize_entity_name(name):
    normalized = re.sub(r"\s+", " ", name.strip().lower())
    normalized = re.sub(r"[^a-z0-9\s\-]", "", normalized)
    return normalized


def resolve_entities(entities):
    buckets = {}

    for entity in entities:
        canonical_key = normalize_entity_name(entity["name"])
        if not canonical_key:
            continue

        if canonical_key not in buckets:
            buckets[canonical_key] = {
                "id": f"entity:{len(buckets)}",
                "name": entity["name"],
                "types": defaultdict(int),
                "aliases": set(),
                "source_claims": set(),
                "clusters": defaultdict(int)
            }

        bucket = buckets[canonical_key]
        bucket["types"][entity["type"]] += 1
        bucket["aliases"].add(entity["name"])
        if entity.get("source_claim"):
            bucket["source_claims"].add(entity["source_claim"])

        cluster_id = entity.get("cluster")
        if cluster_id is not None:
            bucket["clusters"][int(cluster_id)] += 1

    resolved = []
    for bucket in buckets.values():
        top_type = max(bucket["types"].items(), key=lambda item: item[1])[0]
        resolved.append({
            "id": bucket["id"],
            "name": bucket["name"],
            "type": top_type,
            "aliases": sorted(bucket["aliases"]),
            "source_claims": sorted(bucket["source_claims"]),
            "clusters": [
                {"cluster": cluster_id, "mentions": mentions}
                for cluster_id, mentions in sorted(bucket["clusters"].items())
            ],
            "mentions": int(sum(bucket["types"].values()))
        })

    return resolved


def serialize_graph(graph):
    return {
        "nodes": [
            {
                "id": node_id,
                **attrs
            }
            for node_id, attrs in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": source,
                "target": target,
                **attrs
            }
            for source, target, attrs in graph.edges(data=True)
        ]
    }


def build_knowledge_graph(clusters, centroids, resolved_entities):
    graph = nx.Graph()

    for cluster in clusters:
        node_id = f"cluster:{cluster['id']}"
        graph.add_node(
            node_id,
            type="cluster",
            label=f"Cluster {cluster['id']}",
            cluster_id=int(cluster["id"]),
            count=int(cluster["count"]),
            summary=cluster.get("summary", "")
        )

    if len(centroids) > 1:
        similarities = cosine_similarity(centroids)
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                similarity_score = float(similarities[i, j])
                if similarity_score >= CLUSTER_EDGE_SIMILARITY_THRESHOLD:
                    graph.add_edge(
                        f"cluster:{i}",
                        f"cluster:{j}",
                        relation="semantic_similarity",
                        weight=similarity_score
                    )

    for entity in resolved_entities:
        graph.add_node(
            entity["id"],
            type="entity",
            label=entity["name"],
            entity_type=entity["type"],
            mentions=entity["mentions"],
            aliases=entity["aliases"]
        )

        for cluster_ref in entity.get("clusters", []):
            graph.add_edge(
                entity["id"],
                f"cluster:{cluster_ref['cluster']}",
                relation="mentioned_in_claims_for_cluster",
                weight=float(cluster_ref["mentions"])
            )

    return serialize_graph(graph)

# -----------------------------
# MAP CLAIMS TO CLUSTERS
# -----------------------------
def classify_claims(claims, centroids):
    results = []

    for claim in claims:
        claim_embedding = embed_texts([claim])[0]
        sims = cosine_similarity([claim_embedding], centroids)[0]

        best_cluster = np.argmax(sims)
        best_score = sims[best_cluster]

        if best_score < SIMILARITY_THRESHOLD:
            status = "Not represented in corpus"
        else:
            status = f"Aligned with Cluster {best_cluster}"

        results.append({
            "claim": claim,
            "cluster": int(best_cluster),
            "similarity": float(best_score),
            "status": status
        })

    return results

# -----------------------------
# REFLECTIVE OUTPUT
# -----------------------------
def generate_reflection(results):
    print("\n--- Reflective Alignment ---\n")
    for r in results:
        print(f"Claim: {r['claim']}")
        print(f"Status: {r['status']}")
        print(f"Similarity Score: {r['similarity']:.3f}")
        print("")


def prompt_max_results(
    min_results=MIN_RESULTS,
    max_results=MAX_RESULTS,
    default=DEFAULT_RESULTS
):
    while True:
        raw = input(
            f"How many results? ({min_results}-{max_results}, default {default}):\n"
        ).strip()

        if not raw:
            return default

        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if min_results <= value <= max_results:
            return value

        print(f"Please choose a value between {min_results} and {max_results}.")


def build_cluster_payload(
    query,
    rationale,
    explanation="",
    max_results=DEFAULT_RESULTS,
    n_clusters=N_CLUSTERS
):
    require_openai_client()

    pmids = search_pubmed(query, max_results)
    if not pmids:
        raise ValueError("No PubMed records found for that query.")

    corpus = fetch_abstracts(pmids)
    if not corpus:
        raise ValueError("No abstracts with text were retrieved for that query.")

    texts = [paper["abstract"] for paper in corpus]
    embeddings = embed_texts(texts)
    labels, centroids = cluster_embeddings(embeddings, n_clusters=n_clusters)
    points = compute_umap_points(embeddings)

    cluster_docs = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_docs[int(label)].append({
            "title": corpus[idx].get("title", "Untitled"),
            "year": corpus[idx].get("year"),
            "abstract": corpus[idx].get("abstract", "")
        })

    clusters = []
    unique_clusters = sorted(set(int(x) for x in labels.tolist()))
    for cluster_id in unique_clusters:
        summary = summarize_cluster(cluster_id, corpus, labels, rationale=rationale)
        docs = cluster_docs[cluster_id]
        clusters.append({
            "id": cluster_id,
            "count": len(docs),
            "summary": summary,
            "documents": docs[:10]
        })

    analysis_text = (explanation or "").strip() or (rationale or "").strip()
    claims = []
    claim_results = []
    resolved_entities = []

    if analysis_text:
        claims = extract_claims(analysis_text)
        if claims:
            claim_results = classify_claims(claims, centroids)
            claim_cluster_lookup = {}
            for result in claim_results:
                if result["similarity"] >= SIMILARITY_THRESHOLD:
                    claim_cluster_lookup[result["claim"]] = int(result["cluster"])

            extracted_entities = extract_entities_from_claims(claims)
            for entity in extracted_entities:
                source_claim = entity.get("source_claim")
                if source_claim in claim_cluster_lookup:
                    entity["cluster"] = claim_cluster_lookup[source_claim]

            resolved_entities = resolve_entities(extracted_entities)

    knowledge_graph = build_knowledge_graph(clusters, centroids, resolved_entities)

    points_payload = []
    for i, point in enumerate(points):
        points_payload.append({
            "x": float(point[0]),
            "y": float(point[1]),
            "cluster": int(labels[i]),
            "title": corpus[i].get("title", "Untitled")
        })

    with open("corpus.json", "w") as f:
        json.dump(corpus, f, indent=2)

    return {
        "query": query,
        "rationale": rationale,
        "total_documents": len(corpus),
        "clusters": clusters,
        "points": points_payload,
        "claims": claims,
        "claim_alignment": claim_results,
        "entities": resolved_entities,
        "knowledge_graph": knowledge_graph
    }


def create_web_app():
    app = Flask(__name__, static_folder=".")
    CORS(app)

    @app.get("/")
    def index():
        return send_from_directory(".", "index.html")

    @app.get("/demo_payload.json")
    def demo_payload():
        payload_path = Path("demo_payload.json")
        if not payload_path.exists():
            return jsonify({"error": "demo_payload.json not found"}), 404
        return send_from_directory(".", "demo_payload.json")

    @app.get("/demo_payload_human_aging_200.json")
    def demo_payload_human_aging():
        payload_path = Path("demo_payload_human_aging_200.json")
        if not payload_path.exists():
            return jsonify({"error": "demo_payload_human_aging_200.json not found"}), 404
        return send_from_directory(".", "demo_payload_human_aging_200.json")

    @app.post("/api/search")
    def api_search():
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()
        rationale = (payload.get("rationale") or "").strip()
        explanation = (payload.get("explanation") or "").strip()
        max_results = int(payload.get("max_results") or DEFAULT_RESULTS)

        if not query:
            return jsonify({"error": "Please provide a search query."}), 400

        max_results = max(MIN_RESULTS, min(MAX_RESULTS, max_results))

        try:
            result = build_cluster_payload(
                query,
                rationale,
                explanation=explanation,
                max_results=max_results
            )
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app

# -----------------------------
# MAIN PIPELINE
# ---------------------------
def main(no_plot=False):
    if client is None:
        print(
            "OPENAI_API_KEY not found. CLI analysis mode requires OpenAI. "
            "For demo website mode, open index.html directly (portfolio mode)."
        )
        return

    query = input("Enter PubMed search query:\n")
    max_results = prompt_max_results()

    pmids = search_pubmed(query, max_results)
    if not pmids:
        print("No PubMed records found for that query. Try broader terms and run again.")
        return

    corpus = fetch_abstracts(pmids)
    if not corpus:
        print(
            "No abstracts with text were retrieved. Try a broader query or increase result count "
            "within the allowed range."
        )
        return

    # Save locally
    with open("corpus.json", "w") as f:
        json.dump(corpus, f, indent=2)

    print("Embedding abstracts...")
    texts = [paper["abstract"] for paper in corpus]
    embeddings = embed_texts(texts)

    print("Clustering abstracts...")
    labels, centroids = cluster_embeddings(embeddings)
    if no_plot:
        print("Skipping plot (--no-plot enabled)")
    else:
        visualize_clusters(embeddings, labels)

    print("\nCluster Overview:\n")
    for i in range(len(centroids)):
        size = (labels == i).sum()
        print(f"\n=== Cluster {i} ({size} papers) ===")

        summary = summarize_cluster(i, corpus, labels)
        print(summary)

        print("\nRepresentative Papers:")
        titles = [
            corpus[j]["title"]
            for j in range(len(labels))
            if labels[j] == i
        ]
        for title in titles[:3]:
            print(" -", title)

    explanation = input("\nWrite your explanation:\n")

    claims = extract_claims(explanation)
    print("\nExtracted Claims:")
    for c in claims:
        print("-", c)

    results = classify_claims(claims, centroids)
    generate_reflection(results)

    extracted_entities = extract_entities_from_claims(claims)
    claim_cluster_lookup = {
        result["claim"]: int(result["cluster"])
        for result in results
        if result["similarity"] >= SIMILARITY_THRESHOLD
    }
    for entity in extracted_entities:
        source_claim = entity.get("source_claim")
        if source_claim in claim_cluster_lookup:
            entity["cluster"] = claim_cluster_lookup[source_claim]

    resolved_entities = resolve_entities(extracted_entities)

    cluster_summaries = []
    for cluster_id in sorted(set(int(x) for x in labels.tolist())):
        cluster_summaries.append({
            "id": cluster_id,
            "count": int((labels == cluster_id).sum()),
            "summary": ""
        })
    knowledge_graph = build_knowledge_graph(cluster_summaries, centroids, resolved_entities)

    print("\nResolved Entities:")
    for entity in resolved_entities:
        print(f"- {entity['name']} ({entity['type']}) | mentions: {entity['mentions']}")

    print(
        f"\nKnowledge Graph: {len(knowledge_graph['nodes'])} nodes, "
        f"{len(knowledge_graph['edges'])} edges"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Veridian: cluster and analyze PubMed abstracts")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip UMAP cluster visualization window"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Run Veridian as a webpage"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for web mode"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for web mode"
    )
    args = parser.parse_args()
    if args.web:
        app = create_web_app()
        app.run(host=args.host, port=args.port, debug=False)
    else:
        main(no_plot=args.no_plot)