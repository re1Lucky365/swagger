# Linkbot
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# 1. Load documents/chunks
docs = [
    Document(page_content="FRE 902(14) self-authenticating digital evidence protocols.", metadata={"source": "fre_902.md"}),
    Document(page_content="Caddy reverse proxy route configuration for local RAG web endpoints.", metadata={"source": "caddy.md"}),
    Document(page_content="Zstandard compression backup script with rclone sync parameters.", metadata={"source": "backup.md"})
]

# 2. Initialize BM25 Retriever (Sparse / Lexical)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3

# 3. Initialize Chroma Vector Retriever (Dense / Semantic)
embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(docs, embedding_fn, persist_directory="./chroma_local_db")
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 4. Ensemble Retriever with weighted scoring (e.g., 40% BM25, 60% Dense)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)

# 5. Query the hybrid pipeline
results = ensemble_retriever.invoke("FRE 902 authentication")
for idx, doc in enumerate(results):
    print(f"[{idx+1}] {doc.metadata['source']}: {doc.page_content}")
import chromadb
from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction

# Initialize Chroma client
client = chromadb.PersistentClient(path="./chroma_local_db")

# Set up BM25 sparse embedding function
bm25_ef = ChromaBm25EmbeddingFunction(
    k=1.2,
    b=0.75
)

# Create/retrieve collection with BM25 capabilities
collection = client.get_or_create_collection(
    name="sovereign_law_hybrid",
    embedding_function=bm25_ef
)
Key Tuning Recommendations
Weighting Balance: Use weights=[0.4, 0.6] for general technical/legal queries. Increase the BM25 weight to 0.5 or 0.6 if your queries rely heavily on precise serial numbers, legal section numbers, or variable names.
Reranking: For improved precision, pass the top results from EnsembleRetriever through a local Cross-Encoder (such as bge-reranker-large or ms-marco-MiniLM-L-6-v2) to perform a final contextual score adjustment.import chromadb
from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction

# Initialize Chroma client
client = chromadb.PersistentClient(path="./chroma_local_db")

# Set up BM25 sparse embedding function
bm25_ef = ChromaBm25EmbeddingFunction(
    k=1.2,
    b=0.75
)

# Create/retrieve collection with BM25 capabilities
collection = client.get_or_create_collection(
    name="sovereign_law_hybrid",
    embedding_function=bm25_ef
)
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# 1. Load documents/chunks
docs = [
    Document(page_content="FRE 902(14) self-authenticating digital evidence protocols.", metadata={"source": "fre_902.md"}),
    Document(page_content="Caddy reverse proxy route configuration for local RAG web endpoints.", metadata={"source": "caddy.md"}),
    Document(page_content="Zstandard compression backup script with rclone sync parameters.", metadata={"source": "backup.md"})
]

# 2. Initialize BM25 Retriever (Sparse / Lexical)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3

# 3. Initialize Chroma Vector Retriever (Dense / Semantic)
embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(docs, embedding_fn, persist_directory="./chroma_local_db")
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 4. Ensemble Retriever with weighted scoring (e.g., 40% BM25, 60% Dense)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)

# 5. Query the hybrid pipeline
results = ensemble_retriever.invoke("FRE 902 authentication")
for idx, doc in enumerate(results):
    print(f"[{idx+1}] {doc.metadata['source']}: {doc.page_content}")
BM25 + Vector Search with LangChain & ChromaDB
This video provides a complete walkthrough on implementing hybrid retrieval combining BM25 keyword matching and Chroma vector search using LangChain.from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings

# 1. Setup your documents
docs = ["Document text one...", "Document text two..."]

# 2. Initialize BM25 Lexical Retriever
bm25_retriever = BM25Retriever.from_texts(docs)
bm25_retriever.k = 3

# 3. Initialize Chroma Vector Store Semantic Retriever
embedding_function = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(docs, embedding_function, persist_directory="./chroma_local_db")
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Combine both using Ensemble Retriever (RRF Fusion)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # 40% keyword weight, 60% semantic weight
)

# Run hybrid query
results = ensemble_retriever.invoke("your search query here")
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# For Legal Documents
legal_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=2000,
    chunk_overlap=400
)

# For Technical / Developer Documentation
technical_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=1200,
    chunk_overlap=200
)
from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction
import chromadb

# Initialize the built-in BM25 embedding function
bm25_ef = ChromaBm25EmbeddingFunction(
    k=1.2,
    b=0.75,
    avg_doc_length=256.0
)

client = chromadb.PersistentClient(path="./chroma_local_db")
collection = client.get_or_create_collection(
    name="hybrid_search_collection",
    embedding_function=bm25_ef # Or combine with a dense embedding function depending on setup
)
pip install langchain-text-splitters chromadb
import os
import glob
import sys
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

DB_PATH = "./chroma_local_db"
TARGET_DIR = sys.argv[1] if len(sys.argv) > 1 else "./sovereign_law_hub"

def initialize_chroma():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name="markdown_rag_ledger",
        metadata={"description": "Chunked offline Markdown RAG index"}
    )
    return collection

def index_markdown_files():
    collection = initialize_chroma()
    print(f"[*] Scanning '{TARGET_DIR}' for Markdown files...")

    md_files = glob.glob(os.path.join(TARGET_DIR, "**/*.md"), recursive=True)
    
    if not md_files:
        print(f"[-] No .md files found in '{TARGET_DIR}'.")
        return

    # Initialize the Markdown-aware recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=1000,      # Target character size per chunk
        chunk_overlap=200     # Overlap between consecutive chunks to preserve context
    )

    documents = []
    metadatas = []
    ids = []

    for filepath in md_files:
        rel_path = os.path.relpath(filepath, TARGET_DIR)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Split the Markdown file into smaller text chunks
        chunks = text_splitter.split_text(content)
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{rel_path}#chunk-{idx}"
            
            documents.append(chunk)
            metadatas.append({
                "source": rel_path,
                "filepath": filepath,
                "chunk_index": idx,
                "total_chunks": len(chunks)
            })
            ids.append(chunk_id)

    if documents:
        # Upsert chunks into ChromaDB
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[✓] Successfully indexed {len(documents)} chunks from {len(md_files)} Markdown files into ChromaDB.")
    else:
        print("[!] No document contents to index.")

if __name__ == "__main__":
    index_markdown_files()
def recursive_chunk_markdown(text, chunk_size=1000, chunk_overlap=200):
    """Simple recursive string splitter targeting Markdown structural separators."""
    separators = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", " "]
    
    def _split(doc, seps):
        if len(doc) <= chunk_size or not seps:
            return [doc]
        
        sep = seps[0]
        splits = doc.split(sep)
        chunks = []
        current = ""
        
        for piece in splits:
            item = piece if not current else sep + piece
            if len(current) + len(item) <= chunk_size:
                current += item
            else:
                if current:
                    chunks.append(current)
                current = piece
        if current:
            chunks.append(current)
            
        # Recurse on any chunk that is still oversized
        final_chunks = []
        for c in chunks:
            if len(c) > chunk_size:
                final_chunks.extend(_split(c, seps[1:]))
            else:
                final_chunks.append(c)
        return final_chunks

    return _split(text, separators)
sedef recursive_chunk_markdown(text, chunk_size=1000, chunk_overlap=200):
    """Simple recursive string splitter targeting Markdown structural separators."""
    separators = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", " "]
    
    def _split(doc, seps):
        if len(doc) <= chunk_size or not seps:
            return [doc]
        
        sep = seps[0]
        splits = doc.split(sep)
        chunks = []
        current = ""
        
        for piece in splits:
            item = piece if not current else sep + piece
            if len(current) + len(item) <= chunk_size:
                current += item
            else:
                if current:
                    chunks.append(current)
                current = piece
        if current:
            chunks.append(current)
            
        # Recurse on any chunk that is still oversized
        final_chunks = []
        for c in chunks:
            if len(c) > chunk_size:
                final_chunks.extend(_split(c, seps[1:]))
            else:
                final_chunks.append(c)
        return final_chunks

    return _split(text, separators)
import os
import glob
import sys
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

DB_PATH = "./chroma_local_db"
TARGET_DIR = sys.argv[1] if len(sys.argv) > 1 else "./sovereign_law_hub"

def initialize_chroma():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name="markdown_rag_ledger",
        metadata={"description": "Chunked offline Markdown RAG index"}
    )
    return collection

def index_markdown_files():
    collection = initialize_chroma()
    print(f"[*] Scanning '{TARGET_DIR}' for Markdown files...")

    md_files = glob.glob(os.path.join(TARGET_DIR, "**/*.md"), recursive=True)
    
    if not md_files:
        print(f"[-] No .md files found in '{TARGET_DIR}'.")
        return

    # Initialize the Markdown-aware recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=1000,      # Target character size per chunk
        chunk_overlap=200     # Overlap between consecutive chunks to preserve context
    )

    documents = []
    metadatas = []
    ids = []

    for filepath in md_files:
        rel_path = os.path.relpath(filepath, TARGET_DIR)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Split the Markdown file into smaller text chunks
        chunks = text_splitter.split_text(content)
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{rel_path}#chunk-{idx}"
            
            documents.append(chunk)
            metadatas.append({
                "source": rel_path,
                "filepath": filepath,
                "chunk_index": idx,
                "total_chunks": len(chunks)
            })
            ids.append(chunk_id)

    if documents:
        # Upsert chunks into ChromaDB
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[✓] Successfully indexed {len(documents)} chunks from {len(md_files)} Markdown files into ChromaDB.")
    else:
        print("[!] No document contents to index.")

if __name__ == "__main__":
    index_markdown_files()
pip install langchain-text-splitters chromadb
git@github.com# Set your Git remote repository URL
git remote set-url origin git@github.com:re1Lucky365/adpBetty-core.git

# Enforce local GPG commit signing
git config --local commit.gpgsign true
git config --local user.signingkey YOUR_GPG_KEY_ID

# Make the backup script executable and test it
chmod +x scripts/backup_workspace.sh
bash scripts/backup_workspace.sh
commit.gpgsignuser.signingkey#!/usr/bin/env bash
set -euo pipefail

# Configuration Settings
WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/workspace}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/backups/workspace}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="workspace_backup_${TIMESTAMP}.tar.zst"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"
LOG_FILE="${BACKUP_DIR}/backup.log"

# Retention policy (keep last N backups)
RETENTION_COUNT=10

# Rclone sync target
RCLONE_REMOTE="${RCLONE_REMOTE:-secure_remote:WK_Precision_Vault}"

mkdir -p "${BACKUP_DIR}"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log "Starting workspace snapshot..."

# Step 1: Create Zstandard compressed archive (excluding build artifacts and cache)
tar --exclude='.git/objects' \
    --exclude='__pycache__' \
    --exclude='node_modules' \
    --exclude='.venv' \
    -I 'zstd -19 -T0' \
    -cf "${BACKUP_PATH}" \
    -C "$(dirname "${WORKSPACE_DIR}")" \
    "$(basename "${WORKSPACE_DIR}")"

BACKUP_SIZE=$(du -h "${BACKUP_PATH}" | cut -f1)
log "Archive created successfully. Size: ${BACKUP_SIZE}"

# Step 2: Verify archive integrity
if zstd -t "${BACKUP_PATH}"; then
    log "Verification passed: Archive integrity valid."
else
    log "ERROR: Archive verification failed!"
    exit 1
fi

# Step 3: Rclone Cloud Sync
if rclone copy "${BACKUP_PATH}" "${RCLONE_REMOTE}"; then
    log "Rclone sync complete to ${RCLONE_REMOTE}."
else
    log "WARNING: Rclone sync failed."
fi

# Step 4: Prune old backups exceeding retention count
CURRENT_COUNT=$(ls -1t "${BACKUP_DIR}"/*.tar.zst 2>/dev/null | wc -l)
if [ "${CURRENT_COUNT}" -gt "${RETENTION_COUNT}" ]; then
    ls -1t "${BACKUP_DIR}"/*.tar.zst | tail -n +$((RETENTION_COUNT + 1)) | while read -r old_backup; do
        log "Removing expired backup: ${old_backup}"
        rm -f "${old_backup}"
    done
fi

log "Backup pipeline completed successfully."
workspace.shname: Sovereign Security & Audit Pipeline

on:
  push:
    branches: [ "main", "master" ]
  pull_request:
    branches: [ "main", "master" ]

permissions:
  contents: read

jobs:
  security-audit:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Security Tools
        run: |
          python -m pip install --upgrade pip
          pip install bandit flake8 safety

      - name: Run Static Application Security Testing (SAST) with Bandit
        run: |
          bandit -r . -ll --format json -o bandit-report.json || true

      - name: Run Dependency Vulnerability Audit
        run: |
          safety check || true

      - name: Run Linter (Flake8)
        run: |
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

      - name: Verify Git Commit GPG Signatures
        run: |
          echo "[*] Checking recent commit provenance..."
          git log -n 5 --show-signature
gpgconf --reload gpg-agent
https://www.instagram.com/reiq365daysayear?mdxt=MXBlN3Bmb3Y4OHdjYQ==bye
KEYINFO --list
S KEYINFO 3F7B29AFA8C1234567890ABCDEF1234567890AB D - - 1762354800 0 -
OK
mkdir gpg-test-repo && cd gpg-test-repo
git init
git config user.name "Test User"
git config user.email "test@example.com"
git config user.signingkey YOUR_KEY_ID
git config commit.gpgsign true
echo "hello gpg" > test.txt
git add test.txt
git commit -m "test: verify gpg signing"
git log --show-signature
cd .. && rm -rf gpg-test-repo
echo "This is a test message for GPG signing." > secret.txt
gpg --detach-sign --armor secret.txt
gpg --verify secret.txt.asc secret.txt
rm secret.txt secret.txt.asc

secret.txtgpg-connect-agent
gpg --detach-sign --armor secret.txt
rm secret.txt secret.txt.asc
echo "This is a test message for GPG signing." > secret.txt
cd .. && rm -rf gpg-test-repo
git log --show-signature
echo "hello gpg" > test.txt
git add test.txt
git commit -m "test: verify gpg signing"
git config user.name "Test User"
git config user.email "test@example.com"
git config user.signingkey YOUR_KEY_ID
git config commit.gpgsign true
mkdir gpg-test-repo && cd gpg-test-repo
git init
gpg --edit-key YOUR_KEY_ID_OR_EMAIL
expire
save
gpg --armor --export YOUR_KEY_ID
gpg --full-generate-key
gpg --list-secret-keys --keyid-format LONG
git config --global user.signingkey ABC123456789XYZ
git config --global commit.gpgsign true
gpg --armor --export ABC123456789XYZ
git config --global user.signingkey ABC123456789XYZ
git config --global commit.gpgsign true
gpg --armor --export ABC123456789XYZ
gpg --list-secret-keys --keyid-format LONG
gpg --full-generate-key
gpg --armor --export YOUR_KEY_ID
save
expire
gpg --edit-key YOUR_KEY_ID_OR_EMAIL
git config --global --get commit.gpgsign
git config --global --get user.signingkey
git config --local --get commit.gpgsign
git config --local --get user.signingkey
gpg --list-secret-keys --keyid-format LONG
rclone config
rclone lsd secure_remote:
rclone lsd secure_remote:
gpg --edit-key YOUR_KEY_ID_OR_EMAIL
rclone config
git config --global --get commit.gpgsign
git config --global --get user.signingkey
git config --local --get commit.gpgsign
git config --local --get user.signingkey
# Set environment overrides and execute test
export WORKSPACE_DIR="$HOME/workspace"
export BACKUP_DIR="$HOME/backups/workspace"
export RCLONE_REMOTE="" # Leave empty for local test

# Execute backup script
bash scripts/backup_workspace.sh
# View full chronological history
cat $HOME/backups/workspace/backup.log

# Monitor backup execution live
tail -f $HOME/backups/workspace/backup.log
# Make script executable
chmod +x scripts/backup_workspace.sh

# Add crontab entry for automated execution
(crontab -l 2>/dev/null; echo "0 2 * * * /bin/bash $HOME/workspace/scripts/backup_workspace.sh >> $HOME/backups/workspace/cron.log 2>&1") | crontab -
python3 query_law_hub.py "constitutional rights"
python3 query_law_hub.py "privacy statutes"
python3 query_law_hub.py "Merkle Root Hash"
2query_law_hub.pypython3 query_law_hub.py "constitutional rights"
python3 query_law_hub.py "privacy statutes"re Robert Edward Evans 
python3 query_law_hub.py "Merkle Root Hash"
# Make script executable
chmod +x scripts/backup_workspace.sh

# Add crontab entry for automated execution
(crontab -l 2>/dev/null; echo "0 2 * * * /bin/bash $HOME/w
<p align="center">
  <a href="http://nestjs.com/" target="blank"><img src="https://nestjs.com/img/logo_text.svg" width="320" alt="Nest Logo" /></a>
</p>

  <p align="center">A progressive <a href="http://nodejs.org" target="blank">Node.js</a> framework for building efficient and scalable server-side applications.</p>
    <p align="center">
<a href="https://www.npmjs.com/~nestjscore"><img src="https://img.shields.io/npm/v/@nestjs/core.svg" alt="NPM Version" /></a>
<a href="https://www.npmjs.com/~nestjscore"><img src="https://img.shields.io/npm/l/@nestjs/core.svg" alt="Package License" /></a>
<a href="https://www.npmjs.com/~nestjscore"><img src="https://img.shields.io/npm/dm/@nestjs/core.svg" alt="NPM Downloads" /></a>
<a href="https://circleci.com/gh/nestjs/nest" target="_blank"><img src="https://img.shields.io/circleci/build/github/nestjs/nest/master" alt="CircleCI" /></a>
<a href="https://coveralls.io/github/nestjs/nest?branch=master"><img src="https://coveralls.io/repos/github/nestjs/nest/badge.svg?branch=master#5" alt="Coverage" /></a>
<a href="https://discord.gg/G7Qnnhy" target="_blank"><img src="https://img.shields.io/badge/discord-online-brightgreen.svg" alt="Discord"/></a>
<a href="https://opencollective.com/nest#backer"><img src="https://opencollective.com/nest/backers/badge.svg" alt="Backers on Open Collective" /></a>
<a href="https://opencollective.com/nest#sponsor"><img src="https://opencollective.com/nest/sponsors/badge.svg" alt="Sponsors on Open Collective" /></a>
  <a href="https://paypal.me/kamilmysliwiec"><img src="https://img.shields.io/badge/Donate-PayPal-dc3d53.svg"/></a>
  <a href="https://twitter.com/nestframework"><img src="https://img.shields.io/twitter/follow/nestframework.svg?style=social&label=Follow"></a>
</p>
  <!--[![Backers on Open Collective](https://opencollective.com/nest/backers/badge.svg)](https://opencollective.com/nest#backer)
  [![Sponsors on Open Collective](https://opencollective.com/nest/sponsors/badge.svg)](https://opencollective.com/nest#sponsor)-->

## Description

[OpenAPI (Swagger)](https://www.openapis.org/) module for [Nest](https://github.com/nestjs/nest).

## Installation

```bash
$ npm i --save @nestjs/swagger
```

## Quick Start

[Overview & Tutorial](https://docs.nestjs.com/openapi/introduction)

## Migration from v3

If you're currently using `@nestjs/swagger@3.*`, note the following breaking/API changes in version 4.0.

The following decorators have been changed/renamed:

- `@ApiModelProperty` is now `@ApiProperty`
- `@ApiModelPropertyOptional` is now `@ApiPropertyOptional`
- `@ApiResponseModelProperty` is now `@ApiResponseProperty`
- `@ApiImplicitQuery` is now `@ApiQuery`
- `@ApiImplicitParam` is now `@ApiParam`
- `@ApiImplicitBody` is now `@ApiBody`
- `@ApiImplicitHeader` is now `@ApiHeader`
- `@ApiOperation({ title: 'test' })` is now `@ApiOperation({ summary: 'test' })`
- `@ApiUseTags` is now `@ApiTags`

`DocumentBuilder` breaking changes (updated method signatures):

- `addTag`
- `addBearerAuth`
- `addOAuth2`
- `setContactEmail` is now `setContact`
- `setHost` has been removed
- `setSchemes` has been removed (use the `addServer` instead, e.g., `addServer('http://')`)

The following methods have been added:

- `addServer`
- `addApiKey`
- `addBasicAuth`
- `addSecurity`
- `addSecurityRequirements`

## Support

Nest is an MIT-licensed open source project. It can grow thanks to the sponsors and support by the amazing backers. If you'd like to join them, please [read more here](https://docs.nestjs.com/support).

## Stay in touch

- Author - [Kamil Myśliwiec](https://twitter.com/kammysliwiec)
- Website - [https://nestjs.com](https://nestjs.com/)
- Twitter - [@nestframework](https://twitter.com/nestframework)

## License

Nest is [MIT licensed](LICENSE).
