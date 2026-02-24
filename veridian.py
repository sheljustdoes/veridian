import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import umap
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


def load_environment():
    for env_filename in (".env", ".ENV"):
        env_path = Path(env_filename)
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"Loaded environment from {env_filename}")
            return

    load_dotenv()


load_environment()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found. Add it to .env/.ENV or export it in your shell.")

client = OpenAI()

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


def prompt_max_results(min_results=25, max_results=50, default=50):
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


def build_cluster_payload(query, rationale, max_results=50, n_clusters=N_CLUSTERS):
    pmids = search_pubmed(query, max_results)
    if not pmids:
        raise ValueError("No PubMed records found for that query.")

    corpus = fetch_abstracts(pmids)
    if not corpus:
        raise ValueError("No abstracts with text were retrieved for that query.")

    texts = [paper["abstract"] for paper in corpus]
    embeddings = embed_texts(texts)
    labels, _ = cluster_embeddings(embeddings, n_clusters=n_clusters)
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
        "points": points_payload
    }


def create_web_app():
    app = Flask(__name__, static_folder=".")
    CORS(app)

    @app.get("/")
    def index():
        return send_from_directory(".", "index.html")

    @app.post("/api/search")
    def api_search():
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()
        rationale = (payload.get("rationale") or "").strip()
        max_results = int(payload.get("max_results") or 50)

        if not query:
            return jsonify({"error": "Please provide a search query."}), 400

        max_results = max(25, min(50, max_results))

        try:
            result = build_cluster_payload(query, rationale, max_results=max_results)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app

# -----------------------------
# MAIN PIPELINE
# ---------------------------
def main(no_plot=False):
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