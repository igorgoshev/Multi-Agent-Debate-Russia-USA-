import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def load_usa_embeddings():
    """Load USA embeddings into Qdrant collection"""

    # Initialize Qdrant client
    client = QdrantClient(host="localhost", port=6333)

    # Load USA embeddings
    print("Loading USA embeddings...")
    with open("agent_usa_embeddings.json", 'r', encoding='utf-8') as f:
        embeddings_data = json.load(f)

    print(f"Found {len(embeddings_data)} USA chunks")

    # Get vector dimension
    vector_size = len(embeddings_data[0]['embedding'])
    print(f"Vector dimension: {vector_size}")

    # Create USA collection
    collection_name = "usa_collection"
    try:
        client.get_collection(collection_name)
        print(f"Collection {collection_name} already exists")
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created collection: {collection_name}")

    # Prepare points for upload
    points = []
    for i, item in enumerate(embeddings_data):
        points.append(PointStruct(
            id=i,  # Use simple integer ID
            vector=item['embedding'],
            payload={
                'content': item['content'],
                'chunk_id': item['chunk_id']  # Keep original ID in payload
            }
        ))

    # Upload to Qdrant in batches
    batch_size = 500
    total_batches = (len(points) + batch_size - 1) // batch_size

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

        current_batch = i // batch_size + 1
        progress = (current_batch / total_batches) * 100
        print(f"📦 Uploaded batch {current_batch}/{total_batches} ({progress:.1f}%)")

    print(f"✅ Successfully loaded {len(points)} USA embeddings!")


# Run it
if __name__ == "__main__":
    load_usa_embeddings()