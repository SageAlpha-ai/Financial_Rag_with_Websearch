# Deployment Checklist

Use this checklist to ensure your deployment is complete and working.

## Pre-Deployment

- [ ] All code committed to GitHub
- [ ] `.env` file is NOT committed (in `.gitignore`)
- [ ] `render.yaml` and `Procfile` are committed
- [ ] `requirements.txt` is up to date
- [ ] `runtime.txt` specifies Python 3.11

## Environment Variables Setup

### Azure OpenAI
- [ ] `AZURE_OPENAI_API_KEY` - Set in Render
- [ ] `AZURE_OPENAI_ENDPOINT` - Set in Render
- [ ] `AZURE_OPENAI_API_VERSION` - Set in Render (or use default)
- [ ] `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` - Set in Render
- [ ] `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME` - Set in Render

### Azure Blob Storage
- [ ] `AZURE_STORAGE_CONNECTION_STRING` - Set in Render
- [ ] `AZURE_BLOB_CONTAINER_NAME` - Set in Render

### Chroma Cloud
- [ ] `CHROMA_API_KEY` - Set in Render
- [ ] `CHROMA_TENANT` - Set in Render
- [ ] `CHROMA_DATABASE` - Set in Render
- [ ] `CHROMA_COLLECTION_NAME` - Set in Render (or use default)
- [ ] `CHROMA_AUTO_CREATE_COLLECTION` - Set to `true` in Render

### Optional
- [ ] `SERP_API_KEY` - Set in Render (if using web search)
- [ ] `RAG_API_KEY` - Set in Render (if using API auth)
- [ ] `CORS_ORIGINS` - Set in Render (if needed)

## Deployment Steps

- [ ] Created Render account
- [ ] Connected GitHub repository
- [ ] Created new Web Service
- [ ] Added all environment variables
- [ ] Service deployed successfully
- [ ] Service is "Live" in Render dashboard

## Post-Deployment Verification

- [ ] Health check works: `curl https://your-app.onrender.com/health`
- [ ] Swagger UI loads: `https://your-app.onrender.com/docs`
- [ ] API responds to test query
- [ ] Logs show no errors
- [ ] Azure OpenAI deployments validated (check logs)

## Ingestion

- [ ] Connected to Render Shell
- [ ] Ran `python ingest.py --fresh`
- [ ] Ingestion completed successfully
- [ ] Documents stored in Chroma Cloud
- [ ] Verified document count in logs

## Testing

- [ ] Test query endpoint: `POST /query`
- [ ] Verify response format
- [ ] Check answer_type is correct
- [ ] Verify sources are included
- [ ] Test with different query types

## Production Readiness

- [ ] Set up monitoring/alerts
- [ ] Configure custom domain (optional)
- [ ] Set up API authentication (if needed)
- [ ] Review and optimize costs
- [ ] Document API endpoints for team
- [ ] Set up backup strategy for Chroma data

## Common Issues

### Service won't start
- [ ] Check all environment variables are set
- [ ] Verify Azure OpenAI deployment names are correct
- [ ] Check logs for specific error messages

### Health check fails
- [ ] Verify Chroma Cloud credentials
- [ ] Check Azure OpenAI API key is valid
- [ ] Ensure all required env vars are present

### Slow responses
- [ ] Check if service is on free tier (spins down)
- [ ] Consider upgrading to paid plan
- [ ] Optimize query complexity

### Ingestion fails
- [ ] Verify Azure Blob Storage connection string
- [ ] Check container name is correct
- [ ] Verify Chroma Cloud credentials
- [ ] Check network connectivity from Render

## Success Criteria

✅ Service is live and accessible  
✅ Health endpoint returns 200  
✅ Query endpoint returns valid responses  
✅ Documents are ingested and searchable  
✅ Logs show no critical errors  
✅ API documentation is accessible  

## Next Steps After Deployment

1. Share API URL with team
2. Set up monitoring
3. Configure custom domain (optional)
4. Set up CI/CD for auto-deployment
5. Document API usage for consumers
