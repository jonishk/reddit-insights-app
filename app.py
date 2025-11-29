import pinecone
print("=== DEBUG PINECONE CLIENT ===")
print("module =", pinecone)
print("dir =", dir(pinecone))
print("Pinecone =", getattr(pinecone, "Pinecone", None))

try:
    print("Initializing Pinecone client...")
    print("API Key present:", bool(PINECONE_API_KEY))

    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("pc object =", pc)

    print("Listing indexes...")
    existing = pc.list_indexes()
    print("Indexes returned:", existing)

except Exception as e:
    print("CRITICAL PINECONE INIT ERROR:", repr(e))
    pc = None


