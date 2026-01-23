#!/bin/bash
# Azure Deployment Script
# Run this script to deploy to Azure App Service

set -e

echo "=========================================="
echo "Azure App Service Deployment"
echo "=========================================="

# Configuration
APP_NAME="${AZURE_WEBAPP_NAME:-financial-rag-api}"
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-financial-rag-rg}"
LOCATION="${AZURE_LOCATION:-eastus}"
RUNTIME="PYTHON:3.11"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "Error: Azure CLI is not installed"
    echo "Install from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Login to Azure
echo "Logging in to Azure..."
az login

# Create resource group if it doesn't exist
echo "Creating resource group: $RESOURCE_GROUP"
az group create --name $RESOURCE_GROUP --location $LOCATION || true

# Create App Service plan if it doesn't exist
PLAN_NAME="${APP_NAME}-plan"
echo "Creating App Service plan: $PLAN_NAME"
az appservice plan create \
    --name $PLAN_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku B1 \
    --is-linux || true

# Create Web App
echo "Creating Web App: $APP_NAME"
az webapp create \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --plan $PLAN_NAME \
    --runtime $RUNTIME || true

# Configure app settings
echo "Configuring application settings..."
az webapp config appsettings set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
        SCM_DO_BUILD_DURING_DEPLOYMENT=true \
        ENABLE_ORYX_BUILD=true \
        SCM_COMMAND_IDLE_TIMEOUT=600 \
        WEBSITE_TIME_ZONE=UTC || true

# Set startup command
echo "Setting startup command..."
az webapp config set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --startup-file "gunicorn api:app --bind 0.0.0.0:8000 --workers 2 --timeout 300 --worker-class uvicorn.workers.UvicornWorker" || true

# Deploy code
echo "Deploying code..."
az webapp up \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --runtime $RUNTIME

echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo "App URL: https://${APP_NAME}.azurewebsites.net"
echo ""
echo "Next steps:"
echo "1. Configure environment variables in Azure Portal"
echo "2. Check logs: az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo "3. Test health endpoint: https://${APP_NAME}.azurewebsites.net/health"
