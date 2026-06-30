# TFG-RAG-VDB
Configuración docker local del Chromadb:
```
docker run -v ./chroma-data:/data -v ./config.yaml:/config.yaml -p 8000:8000 chromadb/chroma                 
```