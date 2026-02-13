# Azure Event Grid Setup for Automated Ingestion

This guide explains how to set up Azure Event Grid to automatically trigger document ingestion when new files are added to Azure Blob Storage.

## Overview

When a data engineer adds documents to Azure Blob Storage, Azure Event Grid will automatically call your FastAPI application's `/ingest/webhook` endpoint, which triggers incremental ingestion.

## Prerequisites

- Azure Storage Account with Blob Storage
- FastAPI application deployed (Azure App Service, Container Instances, etc.)
- Public HTTPS endpoint for your FastAPI app

## Step 1: Get Your Webhook URL

Your FastAPI application must be accessible via HTTPS. The webhook URL will be:

```
https://<your-app-name>.azurewebsites.net/ingest/webhook
```

Or if using a custom domain:

```
https://<your-domain>/ingest/webhook
```

**Important:** The endpoint must be publicly accessible (no authentication required, or you must configure Event Grid authentication).

## Step 2: Configure Azure Event Grid

### Option A: Using Azure Portal (Recommended)

1. **Navigate to Storage Account**
   - Go to [Azure Portal](https://portal.azure.com)
   - Find your Storage Account
   - Click on **Events** in the left menu

2. **Create Event Subscription**
   - Click **+ Event Subscription**
   - Fill in the form:
     - **Name:** `blob-ingestion-webhook`
     - **Event Schema:** Event Grid Schema
     - **System Topic Name:** (auto-generated or custom)
   
3. **Configure Event Types**
   - Click **Filter to Event Types**
   - Select:
     - ✅ **Blob Created** (when new files are uploaded)
     - ✅ **Blob Deleted** (optional, for cleanup)
   - Uncheck other event types

4. **Configure Endpoint**
   - **Endpoint Type:** Web Hook
   - **Endpoint URL:** `https://<your-app-name>.azurewebsites.net/ingest/webhook`
   - Click **Confirm Selection**

5. **Configure Filters (Optional)**
   - **Subject Begins With:** `/blobServices/default/containers/<your-container-name>/`
   - This ensures only events from your specific container trigger ingestion

6. **Create**
   - Click **Create** to save the event subscription

### Option B: Using Azure CLI

```bash
# Set variables
RESOURCE_GROUP="your-resource-group"
STORAGE_ACCOUNT="your-storage-account"
CONTAINER_NAME="your-container-name"
WEBHOOK_URL="https://your-app.azurewebsites.net/ingest/webhook"
EVENT_SUBSCRIPTION_NAME="blob-ingestion-webhook"

# Get storage account resource ID
STORAGE_ID=$(az storage account show \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --query id \
  --output tsv)

# Create event subscription
az eventgrid event-subscription create \
  --name $EVENT_SUBSCRIPTION_NAME \
  --source-resource-id $STORAGE_ID \
  --endpoint-type webhook \
  --endpoint $WEBHOOK_URL \
  --included-event-types Microsoft.Storage.BlobCreated \
  --subject-begins-with "/blobServices/default/containers/$CONTAINER_NAME/"
```

### Option C: Using Azure PowerShell

```powershell
# Set variables
$resourceGroup = "your-resource-group"
$storageAccount = "your-storage-account"
$containerName = "your-container-name"
$webhookUrl = "https://your-app.azurewebsites.net/ingest/webhook"
$subscriptionName = "blob-ingestion-webhook"

# Get storage account resource ID
$storageId = (Get-AzStorageAccount `
  -ResourceGroupName $resourceGroup `
  -Name $storageAccount).Id

# Create event subscription
New-AzEventGridSubscription `
  -EventSubscriptionName $subscriptionName `
  -ResourceId $storageId `
  -Endpoint $webhookUrl `
  -IncludedEventType Microsoft.Storage.BlobCreated `
  -SubjectBeginsWith "/blobServices/default/containers/$containerName/"
```

## Step 3: Test the Webhook

### Manual Test

1. **Upload a test file to Azure Blob Storage**
   ```bash
   az storage blob upload \
     --account-name your-storage-account \
     --container-name your-container \
     --name test-document.pdf \
     --file ./test-document.pdf
   ```

2. **Check ingestion logs**
   - View your FastAPI application logs
   - You should see:
     ```
     [WEBHOOK] Received event: Microsoft.Storage.BlobCreated for /blobServices/default/containers/...
     [INGESTION JOB ...] Starting incremental ingestion
     ```

3. **Check ingestion status**
   ```bash
   curl https://your-app.azurewebsites.net/ingest/jobs
   ```

### Verify Event Grid Delivery

1. Go to **Event Grid** → **Event Subscriptions** → `blob-ingestion-webhook`
2. Click **Metrics** tab
3. Check:
   - **Published Events:** Should increase when blobs are created
   - **Matched Events:** Should match published events
   - **Delivery Succeeded:** Should be 100% (or close)

## Step 4: Monitor Ingestion

### Check Job Status

```bash
# List all ingestion jobs
curl https://your-app.azurewebsites.net/ingest/jobs

# Check specific job status
curl https://your-app.azurewebsites.net/ingest/status/<job_id>
```

### Application Logs

View logs in Azure Portal:
- **App Service** → **Log stream**
- Or **Application Insights** (if configured)

Look for:
- `[WEBHOOK] Received event: ...`
- `[INGESTION JOB ...] Starting incremental ingestion`
- `[INGESTION JOB ...] Completed: X documents`

## Step 5: Security (Optional but Recommended)

### Option 1: API Key Authentication

Add API key validation to the webhook endpoint:

1. **Set API key in environment variables:**
   ```bash
   INGESTION_WEBHOOK_KEY=your-secret-key-here
   ```

2. **Update webhook endpoint** to validate the key:
   ```python
   # In app.py
   webhook_key = Header(None, alias="X-API-Key")
   
   if webhook_key != os.getenv("INGESTION_WEBHOOK_KEY"):
       raise HTTPException(status_code=401, detail="Invalid API key")
   ```

3. **Configure Event Grid with custom headers:**
   - Azure Portal → Event Subscription → **Advanced Features**
   - Add custom header: `X-API-Key: your-secret-key-here`

### Option 2: Azure AD Authentication

For production, consider using Azure AD authentication with managed identity.

## Troubleshooting

### Webhook Not Receiving Events

1. **Check Event Grid subscription status:**
   - Azure Portal → Event Grid → Event Subscriptions
   - Status should be **Active**

2. **Verify endpoint URL:**
   - Test the endpoint manually:
     ```bash
     curl -X POST https://your-app.azurewebsites.net/ingest/webhook \
       -H "Content-Type: application/json" \
       -d '{"eventType": "test"}'
     ```

3. **Check Event Grid delivery logs:**
   - Event Subscription → **Metrics**
   - Look for **Delivery Failed** events
   - Check **Dead Letter** for failed deliveries

### Ingestion Not Processing Documents

1. **Check application logs:**
   - Look for errors in ingestion pipeline
   - Verify Azure Blob Storage connection
   - Verify Chroma Cloud connection

2. **Test incremental ingestion manually:**
   ```bash
   curl -X POST https://your-app.azurewebsites.net/ingest/incremental
   ```

3. **Check processed blobs tracking:**
   - The system tracks processed blobs to avoid duplicates
   - If a blob was already processed, it won't be re-processed
   - Use `/ingest?fresh=true` to force re-processing

### High Latency

- Ingestion runs in background tasks
- Large documents may take time to process
- Check job status endpoint for progress

## Alternative: Scheduled Ingestion

If Event Grid is not suitable, you can set up scheduled ingestion using:

### Azure Logic Apps
- Create a Logic App with a scheduled trigger
- Call `/ingest/incremental` endpoint on schedule

### Azure Functions Timer Trigger
- Create an Azure Function with a timer trigger
- Call your FastAPI `/ingest/incremental` endpoint

### Cron Job (if using Azure Container Instances)
- Add a cron job to your container
- Run: `curl -X POST http://localhost:8000/ingest/incremental`

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Full ingestion (all documents) |
| `/ingest/incremental` | POST | Incremental ingestion (new/updated only) |
| `/ingest/webhook` | POST | Azure Event Grid webhook handler |
| `/ingest/status/{job_id}` | GET | Get ingestion job status |
| `/ingest/jobs` | GET | List all ingestion jobs |

## Example: Complete Setup Script

```bash
#!/bin/bash

# Configuration
RESOURCE_GROUP="financial-rag-rg"
STORAGE_ACCOUNT="financialragstorage"
CONTAINER_NAME="documents"
APP_URL="https://financial-rag-app.azurewebsites.net"
WEBHOOK_URL="$APP_URL/ingest/webhook"

# Create event subscription
az eventgrid event-subscription create \
  --name blob-ingestion-automation \
  --source-resource-id $(az storage account show \
    --name $STORAGE_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --query id -o tsv) \
  --endpoint-type webhook \
  --endpoint $WEBHOOK_URL \
  --included-event-types Microsoft.Storage.BlobCreated \
  --subject-begins-with "/blobServices/default/containers/$CONTAINER_NAME/"

echo "✅ Event Grid subscription created!"
echo "📝 Webhook URL: $WEBHOOK_URL"
echo "🧪 Test by uploading a file to: $CONTAINER_NAME container"
```

## Next Steps

1. ✅ Set up Event Grid subscription
2. ✅ Test with a sample document
3. ✅ Monitor ingestion jobs
4. ✅ Configure alerts for failed ingestions (optional)
5. ✅ Set up Application Insights for detailed monitoring (optional)

---

**Need Help?**
- Check application logs: `az webapp log tail --name <app-name> --resource-group <rg>`
- View Event Grid metrics in Azure Portal
- Test endpoints manually using Swagger UI: `https://your-app.azurewebsites.net/docs`
