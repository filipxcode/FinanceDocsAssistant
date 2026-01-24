import os
from sentence_transformers import SentenceTransformer

# Tworzymy folder na modele
os.makedirs("local_models", exist_ok=True)

print("⏳ Pobieranie modelu BAAI/bge-m3... To może chwilę potrwać.")
# To pobierze model i zapisze go w folderze 'local_models/bge-m3'
model = SentenceTransformer("BAAI/bge-m3")
model.save("local_models/bge-m3")

print("✅ Model pobrany do folderu local_models/bge-m3")