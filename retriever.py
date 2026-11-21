def retrieve_precedents(query_text: str, top_k: int = 3) -> list[dict]:
    """
    Args:
        query_text: Combined ticket subject and body
        top_k: Number of historical tickets to return (default: 3)
        
    Returns:
        list of dicts, each containing:
            "ticket_id": str (e.g. "TKT-00123")
            "similarity": float (cosine similarity between 0.0 and 1.0)
            "historical_answer": str (the past resolution text from 'answer' column)
    """
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(
        "BAAI/bge-m3",
        device=device,
        model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32},
    )
    q_emb = embedder.encode([query_text], normalize_embeddings=True)

    import chromadb
    client = chromadb.PersistentClient(path="./chroma_dbs")
    col = client.get_collection("my_data")
    res = col.query(
        query_embeddings=q_emb.tolist(),
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    results = []
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        similarity = max(0.0, min(1.0, 1.0 - dist))   # Chroma cosine distance = 1 - cosine similarity
        results.append({
            "ticket_id": str(meta["ticket_id"]),
            "similarity": round(float(similarity), 4),
            "historical_answer": str(meta["answer"]),
        })
    return results

