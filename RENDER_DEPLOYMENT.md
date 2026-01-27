# Render.com Deployment Guide

This guide will help you deploy the Financial RAG API to Render.com in minutes.

## Prerequisites

- GitHub account with your code pushed
- Render.com account (free tier available)
- All required API keys and credentials

## Step-by-Step Deployment

### 1. Push Code to GitHub

```bash
# Initialize git if not already done
git init
git add .
git commit -m "Initial commit - Ready for Render deployment"
git branch -M main
git remote add origin https://github.com/yourusername/your-repo-name.git
git push -u origin main
```

### 2. Create Render Account

1. Go to [render.com](https://render.com)
2. Sign up with GitHub (recommended for easy repo connection)
3. Verify your email

### 3. Create New Web Service

1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository
3. Render will auto-detect the `render.yaml` configuration
4. Service name: `financial-rag-api` (or your preferred name)

### 4. Configure Environment Variables

In Render dashboard, go to **Environment** tab and add:

#### Required Variables:

**Azure OpenAI:**
```
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=text-embedding-3-large
```

**Azure Blob Storage:**
```
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_BLOB_CONTAINER_NAME=your-container-name
```

**Chroma Cloud:**
```
CHROMA_API_KEY=your_chroma_api_key
CHROMA_TENANT=your_tenant_id
CHROMA_DATABASE=your_database_name
CHROMA_COLLECTION_NAME=compliance
```

**SerpAPI (Optional - for web search):**
```
SERP_API_KEY=your_serpapi_key
```

**Optional:**
```
RAG_API_KEY=your_api_key_for_auth  # Leave empty to disable
CHROMA_AUTO_CREATE_COLLECTION=true  # Auto-create collections
CORS_ORIGINS=*  # CORS settings
```

### 5. Deploy

1. Click **"Create Web Service"**
2. Render will:
   - Install Python 3.11
   - Install dependencies from `requirements.txt`
   - Start the service with Gunicorn
3. Wait 2-3 minutes for first deployment

### 6. Verify Deployment

Once deployed, your API will be available at:
- **API**: `https://your-app-name.onrender.com`
- **Swagger UI**: `https://your-app-name.onrender.com/docs`
- **Health Check**: `https://your-app-name.onrender.com/health`

Test with:
```bash
curl https://your-app-name.onrender.com/health
```

### 7. Run Ingestion (First Time)

After deployment, you need to populate the Chroma collection:

**Option A: Using Render Shell**
1. Go to Render dashboard → Your service → **Shell**
2. Run:
   ```bash
   python ingest.py --fresh
   ```

**Option B: Using Render CLI**
```bash
# Install Render CLI
npm install -g render-cli

# Connect to your service
render shell

# Run ingestion
python ingest.py --fresh
```

## Auto-Deployment

Render automatically deploys when you push to your main branch:
- Push to `main` → Auto-deploys
- Push to other branches → No deployment (unless configured)

## Monitoring

- **Logs**: View real-time logs in Render dashboard
- **Metrics**: CPU, Memory, Request count
- **Alerts**: Set up email alerts for service failures

## Troubleshooting

### Service Won't Start

1. **Check Logs**: Render dashboard → Logs tab
2. **Common Issues**:
   - Missing environment variables
   - Wrong deployment names
   - Chroma collection doesn't exist (set `CHROMA_AUTO_CREATE_COLLECTION=true`)

### Health Check Fails

- Verify all environment variables are set
- Check Azure OpenAI deployments exist
- Verify Chroma Cloud credentials

### Slow First Request

- Render free tier spins down after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds
- Upgrade to paid plan for always-on service

## Updating Your Service

1. Make changes locally
2. Commit and push:
   ```bash
   git add .
   git commit -m "Update feature"
   git push origin main
   ```
3. Render auto-deploys (watch in dashboard)

## Cost

- **Free Tier**: 
  - 750 hours/month
  - Spins down after 15 min inactivity
  - Perfect for development/testing
  
- **Starter Plan** ($7/month):
  - Always on
  - Better performance
  - Recommended for production

## Security

- All environment variables are encrypted
- HTTPS enabled by default
- Set `RAG_API_KEY` for API authentication
- Configure `CORS_ORIGINS` for production

## Next Steps

1. ✅ Deploy to Render
2. ✅ Run ingestion
3. ✅ Test API endpoints
4. ✅ Set up monitoring
5. ✅ Configure custom domain (optional)

## Support

- Render Docs: https://render.com/docs
- Render Community: https://community.render.com
- Project Issues: GitHub Issues
