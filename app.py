import pinecone
print("=== DEBUG PINECONE CLIENT ===")
print("module =", pinecone)
print("dir =", dir(pinecone))
print("Pinecone =", getattr(pinecone, "Pinecone", None))
