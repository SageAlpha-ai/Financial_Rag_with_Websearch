# Automated Ingestion System

## Overview

The automated ingestion system automatically processes new documents from Azure Blob Storage when they are added by data engineers. No manual intervention required!

## How It Works

1. **Data Engineer uploads document** → Azure Blob Storage
2. **Azure Event Grid detects** → New blob created
3. **Event Grid calls webhook** → `/ingest/webhook` endpoint
4. **FastAPI triggers ingestion** → Background task processes document
5. **Document is embedded** → Stored in Chroma Cloud
6. **RAG system updated** → New document is searchable

## Quick Start

### 1. Manual Ingestion (One-time Setup)

```bash
# Full ingestion (all documents)
curl -X POST https://your-app.azurewebsites.net/ingest?fresh=true

# Incremental ingestion (new/updated only)
curl -X POST https://your-app.azurewebsites.net/ingest/incremental
```

### 2. Set Up Azure Event Grid (Automation)

Follow the detailed guide: **[AZURE_EVENT_GRID_SETUP.md](./AZURE_EVENT_GRID_SETUP.md)**

**Quick setup:**
```bash
az eventgrid event-subscription create \
  --name blob-ingestion-automation \
  --source-resource-id <storage-account-resource-id> \
  --endpoint-type webhook \
  --endpoint https://your-app.azurewebsites.net/ingest/webhook \
  --included-event-types Microsoft.Storage.BlobCreated
```

### 3. Test Automation

```bash
# Upload a test document
az storage blob upload \
  --account-name your-storage \
  --container-name documents \
  --name test.pdf \
  --file test.pdf

# Check ingestion status
curl https://your-app.azurewebsites.net/ingest/jobs
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Full ingestion (all documents) |
| `/ingest/incremental` | POST | Incremental ingestion (new/updated only) ⭐ **Recommended** |
| `/ingest/webhook` | POST | Azure Event Grid webhook (automatic) |
| `/ingest/status/{job_id}` | GET | Get job status |
| `/ingest/jobs` | GET | List all jobs |

## Features

✅ **Incremental Processing** - Only processes new/updated documents  
✅ **Background Tasks** - Non-blocking, async processing  
✅ **Duplicate Prevention** - Tracks processed blobs (ETag + Last Modified)  
✅ **Error Handling** - Graceful failures, detailed logging  
✅ **Job Tracking** - Monitor ingestion progress  
✅ **Docker Compatible** - Works in containers  

## How Incremental Ingestion Works

1. **Tracks processed blobs** using:
   - Blob name
   - ETag (content hash)
   - Last modified timestamp

2. **Compares with current blobs**:
   - New blob → Process
   - Updated blob (ETag changed) → Re-process
   - Unchanged blob → Skip

3. **Stores metadata** in:
   - Docker: `/tmp/processed_blobs.json`
   - Local: `./processed_blobs.json`
   - Custom: Set `PROCESSED_BLOBS_FILE` env var

## Example Usage

### Manual Trigger (Testing)

```bash
# Trigger incremental ingestion
curl -X POST http://localhost:8000/ingest/incremental

# Response:
{
  "success": true,
  "message": "Incremental ingestion job started",
  "job_id": "incremental_2025-01-27T12:00:00",
  "documents_processed": 0,
  "blobs_processed": 0
}

# Check status
curl http://localhost:8000/ingest/status/incremental_2025-01-27T12:00:00

# Response:
{
  "status": "completed",
  "started_at": "2025-01-27T12:00:00",
  "completed_at": "2025-01-27T12:05:00",
  "documents_processed": 15,
  "blobs_processed": 3,
  "blobs": ["document1.pdf", "document2.xlsx", "document3.txt"]
}
```

### Automated (Production)

Once Event Grid is configured, ingestion happens automatically:

1. Data engineer uploads `financial-report-2024.pdf` to Azure Blob
2. Event Grid triggers webhook → `/ingest/webhook`
3. FastAPI starts background task → Processes document
4. Document is embedded and stored in Chroma
5. RAG system can now answer questions about the new document

## Monitoring

### Check Logs

```bash
# Azure App Service
az webapp log tail --name your-app --resource-group your-rg

# Look for:
# [WEBHOOK] Received event: Microsoft.Storage.BlobCreated
# [INGESTION JOB ...] Starting incremental ingestion
# [INGESTION JOB ...] Completed: 15 documents
```

### View Jobs

```bash
curl https://your-app.azurewebsites.net/ingest/jobs
```

### Application Insights (Optional)

Set up Application Insights for:
- Ingestion success/failure rates
- Processing time metrics
- Error tracking

## Troubleshooting

### Ingestion Not Triggering

1. **Check Event Grid subscription:**
   - Azure Portal → Storage Account → Events
   - Verify subscription is **Active**
   - Check **Metrics** for published events

2. **Test webhook manually:**
   ```bash
   curl -X POST https://your-app.azurewebsites.net/ingest/webhook \
     -H "Content-Type: application/json" \
     -d '{"eventType": "Microsoft.Storage.BlobCreated", "subject": "/blobServices/default/containers/documents/blob.pdf"}'
   ```

3. **Check application logs:**
   - Look for webhook errors
   - Verify Azure Blob connection
   - Verify Chroma Cloud connection

### Documents Not Processing

1. **Check file types:**
   - Supported: `.pdf`, `.xlsx`, `.xls`, `.txt`
   - Unsupported files are skipped

2. **Check blob tracking:**
   - If blob was already processed, it won't re-process
   - Use `fresh=true` to force re-processing

3. **Check job status:**
   ```bash
   curl https://your-app.azurewebsites.net/ingest/jobs
   ```

### High Memory Usage

- Ingestion processes documents in batches (50 by default)
- Large PDFs may consume memory
- Consider increasing container memory limits

## Configuration

### Environment Variables

```bash
# Processed blobs tracking file (optional)
PROCESSED_BLOBS_FILE=/tmp/processed_blobs.json

# Azure Blob Storage (required)
AZURE_STORAGE_CONNECTION_STRING=...
AZURE_BLOB_CONTAINER_NAME=documents

# Chroma Cloud (required)
CHROMA_HOST=...
CHROMA_API_KEY=...
CHROMA_COLLECTION_NAME=default

# Azure OpenAI (required)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=...
```

## Best Practices

1. **Use incremental ingestion** for automation (faster, efficient)
2. **Use full ingestion** only for initial setup or when rebuilding
3. **Monitor job status** regularly to catch failures early
4. **Set up alerts** for failed ingestions (Application Insights)
5. **Test with small files** before processing large documents
6. **Keep processed_blobs.json** backed up (optional, for disaster recovery)

## Security

### Webhook Authentication (Recommended)

Add API key validation to webhook endpoint:

1. Set environment variable:
   ```bash
   INGESTION_WEBHOOK_KEY=your-secret-key
   ```

2. Configure Event Grid with custom header:
   - Azure Portal → Event Subscription → Advanced Features
   - Add header: `X-API-Key: your-secret-key`

3. Update webhook endpoint to validate key (code change required)

## Next Steps

1. ✅ Set up Azure Event Grid (see [AZURE_EVENT_GRID_SETUP.md](./AZURE_EVENT_GRID_SETUP.md))
2. ✅ Test with a sample document
3. ✅ Monitor ingestion jobs
4. ✅ Set up alerts for failures
5. ✅ Document your specific container and storage account names

---

**Questions?** Check the detailed setup guide: [AZURE_EVENT_GRID_SETUP.md](./AZURE_EVENT_GRID_SETUP.md)
