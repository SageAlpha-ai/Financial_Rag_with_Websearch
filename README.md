# SageAlpha AI - Financial RAG with Web Search

Enterprise-grade Financial & Regulatory AI assistant with Hybrid RAG + Web Search capabilities.

## Overview

This is a production-ready FastAPI-based RAG (Retrieval-Augmented Generation) service that provides intelligent, trust-first query answering over financial documents. It combines:

- **Chroma Cloud** for vector storage and retrieval
- **Azure OpenAI** for embeddings and chat completion
- **SerpApi** for live web search and official document retrieval
- **Hybrid RAG + Web Search** for maximum accuracy
- **SageAlpha AI** branding with trust-first guidelines

## Features

- 📄 **Hybrid RAG** - Answers from Azure Blob Storage documents + Live web search
- 🔍 **Web Search Integration** - Automatically retrieves official company investor relations documents
- 🏢 **Company Validation** - Prevents cross-company data contamination
- 🔐 **Trust-First** - Official source citations, no hallucinations
- 📊 **SageAlpha AI Branding** - Professional answer types and source formatting
- ✅ **Never returns "Not available"** - Always provides an answer
- 🌐 **RESTful API** - Ready for Node.js or any HTTP client
- 🚀 **Auto-Deployment** - GitHub Actions → Azure App Service

## Quick Start

### Local Development

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

**Required environment variables:**
- `AZURE_OPENAI_API_KEY` - Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT` - Your Azure OpenAI endpoint
- `AZURE_STORAGE_CONNECTION_STRING` - Azure Blob Storage connection string
- `CHROMA_API_KEY` - Chroma Cloud API key
- `CHROMA_TENANT` - Chroma Cloud tenant ID
- `CHROMA_DATABASE` - Chroma Cloud database name

See `.env.example` for all configuration options.

### 3. Ingest Documents (First Time)

```bash
python ingest.py --fresh
```

This loads documents from Azure Blob Storage and local files, then embeds and stores them in Chroma Cloud.

### 4. Run the API Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Important**: `--host 0.0.0.0` is for server binding (allows external connections). Always use `http://localhost:8000` or `http://127.0.0.1:8000` in your browser - never use `0.0.0.0` in browser URLs.

The API will be available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Usage

### Query Endpoint

The API provides two endpoints for querying:

- **`POST /query`** - Main query endpoint (recommended)
- **`POST /docs/query`** - Alias endpoint for backward compatibility

Both endpoints accept the same request format:

```bash
POST /query
Content-Type: application/json

{
  "query": "What is Oracle Financial Services revenue for FY2023?"
}
```

**Note:** The API accepts either `query` or `question` field (for backward compatibility).

**Response:**

```json
{
  "answer": "Oracle Financial Services Software Ltd reported a revenue of ₹X,XXX crore for FY 2024. This figure is taken from the company's FY 2024 Annual Report, published on its official investor relations website.",
  "answer_type": "sagealpha_ai_search",
  "sources": [
    {
      "title": "Oracle Financial Services – Investor Relations",
      "url": "https://official-site.com/investor-relations",
      "publisher": "Oracle Financial Services Software Ltd"
    }
  ]
}
```

### Node.js Example

```javascript
const response = await fetch("http://localhost:8000/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ 
    query: "What is Oracle Financial Services revenue for FY2023?" 
  })
});

const data = await response.json();
console.log(data.answer);
console.log(data.answer_type); // "sagealpha_rag", "sagealpha_ai_search", or "sagealpha_hybrid_search"
console.log(data.sources); // Array of {title, url, publisher}
```

### Health Check

```bash
GET /health
```

## Deployment

### 🚀 Azure App Service (Recommended)

This project is configured for **automatic deployment to Azure App Service using GitHub Actions**.

#### Quick Setup:

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Create Azure App Service** (see [Azure Deployment Guide](./AZURE_APP_SERVICE_DEPLOYMENT.md))

3. **Configure GitHub Secrets:**
   - Go to GitHub → Settings → Secrets → Actions
   - Add `AZURE_CREDENTIALS` (see [GitHub Deployment Guide](./GITHUB_DEPLOYMENT_GUIDE.md))

4. **Update workflow file:**
   - Edit `.github/workflows/azure-deploy.yml`
   - Change `AZURE_WEBAPP_NAME` to your app name

5. **Push to trigger deployment:**
   ```bash
   git push origin main
   ```

**Full Guide:** [GITHUB_DEPLOYMENT_GUIDE.md](./GITHUB_DEPLOYMENT_GUIDE.md)

### Other Platforms

The service can be deployed to any platform that supports Python:
- Azure App Service (via GitHub Actions) ✅
- Docker (build your own Dockerfile)
- AWS Elastic Beanstalk
- Google Cloud Run
- Heroku

## Project Structure

```
.
├── api.py                 # FastAPI application entry point
├── config/                # Configuration management
│   └── settings.py
├── rag/                   # RAG pipeline logic
│   ├── query_engine.py
│   ├── retriever.py
│   ├── router.py
│   └── answer_formatter.py
├── ingestion/             # Document ingestion
│   ├── azure_blob_loader.py
│   ├── chunking.py
│   └── embed_and_store.py
├── vectorstore/           # Chroma Cloud integration
│   └── chroma_client.py
├── ingest.py             # Ingestion script
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
└── render.yaml           # Render deployment config
```

## Development

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your credentials

# Ingest documents
python ingest.py --fresh

# Run API (use --host 0.0.0.0 for binding, but access via localhost in browser)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Oracle revenue?"}'
```

## 📚 Documentation

- **[DEPLOYMENT_README.md](./DEPLOYMENT_README.md)** - Quick deployment guide
- **[GITHUB_DEPLOYMENT_GUIDE.md](./GITHUB_DEPLOYMENT_GUIDE.md)** - Complete GitHub + Azure guide
- **[AZURE_APP_SERVICE_DEPLOYMENT.md](./AZURE_APP_SERVICE_DEPLOYMENT.md)** - Detailed Azure setup
- **[WEB_SEARCH_INTEGRATION.md](./WEB_SEARCH_INTEGRATION.md)** - Web search integration details
- **[PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)** - Project organization

## 🚀 Deployment

### Quick Deploy (5 minutes)

See **[QUICK_START.md](./QUICK_START.md)** for fastest deployment path.

### GitHub + Azure App Service

This project is configured for **automatic deployment** via GitHub Actions:

1. Push code to GitHub
2. Configure Azure App Service
3. Add GitHub secrets
4. Push to main → Auto-deploys! 🎉

**Full Guide:** [GITHUB_DEPLOYMENT_GUIDE.md](./GITHUB_DEPLOYMENT_GUIDE.md)

## License

Proprietary - All rights reserved
