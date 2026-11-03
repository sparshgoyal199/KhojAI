from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os
import json  # ✅ Import json module

load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

COLLECTION_NAME = "0d7887ca44684c71b407bfd8842a720c"

# 1. Scroll through ALL points
all_chunks = []
offset = None
while True:
    points, offset = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    all_chunks.extend(points)
    
    if offset is None or not points:
        break

print(f"Retrieved {len(all_chunks)} total chunks from the collection.")

# 2. Extract only the payloads (list of dictionaries)
payload_list = []
for chunk in all_chunks:
    if chunk.payload:
        payload_list.append(chunk.payload)

# 3. Define the file path (same directory as this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_file_path = os.path.join(script_dir, "qdrant_payloads.json")

# 4. Write the list of payloads to the JSON file
with open(output_file_path, "w", encoding="utf-8") as f:
    # indent=4 makes the file human-readable; ensure_ascii=False preserves special characters
    json.dump(payload_list, f, indent=4, ensure_ascii=False) 

print(f"✅ Successfully saved all payloads to: {output_file_path}")