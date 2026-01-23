"""
Azure Blob Storage Document Loader

Loads PDF and Excel documents from Azure Blob Storage.
Parses content and extracts structured financial data.
"""

import os
import io
import re
from typing import List, Dict, Optional

from azure.storage.blob import BlobServiceClient, ContainerClient
from pypdf import PdfReader
import pandas as pd

from config.settings import get_config


def get_container_client() -> ContainerClient:
    """Creates Azure Blob Storage container client."""
    config = get_config().azure_blob
    
    client = BlobServiceClient.from_connection_string(config.connection_string)
    return client.get_container_client(config.container_name)


def download_blob(container_client: ContainerClient, blob_name: str) -> bytes:
    """Downloads blob to memory."""
    return container_client.get_blob_client(blob_name).download_blob().readall()


def normalize_fiscal_year(value) -> Optional[str]:
    """
    Normalizes fiscal year to FYxxxx format.
    
    Examples:
    - 2012-03 -> FY2012
    - 2011-12 -> FY2012
    - Mar-2012 -> FY2012
    """
    if pd.isna(value):
        return None
    
    value_str = str(value).strip()
    
    # Match pattern like "2012-03"
    fy_match = re.match(r'(\d{4})-(\d{2})', value_str)
    if fy_match:
        year = fy_match.group(1)
        return f"FY{year}"
    
    # Match 4-digit year
    year_match = re.search(r'(19|20)\d{2}', value_str)
    if year_match:
        year = year_match.group(0)
        return f"FY{year}"
    
    return None


def parse_excel_transposed(content: bytes, blob_name: str) -> List[Dict]:
    """
    Parses TRANSPOSED financial Excel where:
    - Metrics are in rows (column 0)
    - Fiscal years are in columns
    
    Falls back to regular Excel parsing if transposed format not detected.
    """
    documents = []
    excel_file = io.BytesIO(content)
    
    try:
        # Try reading all sheets
        excel_data = pd.read_excel(excel_file, sheet_name=None, header=None)
        
        if not excel_data:
            print("    [WARN] No sheets found in Excel file")
            return documents
        
        print(f"    Found {len(excel_data)} sheet(s)")
        
        # Process each sheet
        for sheet_name, df in excel_data.items():
            print(f"    Processing sheet: {sheet_name} (shape: {df.shape})")
            
            # Find fiscal period row
            fiscal_years = {}
            
            for idx, row in df.iterrows():
                year_cols = []
                for col in range(1, min(len(row), 25)):
                    val = row[col]
                    if pd.notna(val) and re.match(r'\d{4}-\d{2}', str(val)):
                        year_cols.append(col)
                
                if len(year_cols) >= 5:
                    for col in range(1, len(row)):
                        fy = normalize_fiscal_year(row[col])
                        if fy:
                            fiscal_years[col] = fy
                    print(f"    Found fiscal periods: {list(fiscal_years.values())[:5]}...")
                    break
            
            if not fiscal_years:
                print(f"    [WARN] Sheet {sheet_name}: Could not find fiscal period row, trying fallback parsing")
                # Fallback: try to parse as regular Excel
                fallback_docs = parse_excel_fallback(df, blob_name, sheet_name)
                if fallback_docs:
                    documents.extend(fallback_docs)
                    print(f"    [OK] Fallback parsing extracted {len(fallback_docs)} chunks")
                continue
            
            # Find key metrics (only if fiscal_years found)
            metrics = {}
            key_metrics = {
                'revenue': ['revenue'],
                'net_income': ['net income'],
                'ebitda': ['ebitda'],
                'gross_profit': ['gross profit'],
                'operating_income': ['operating income', 'ebit'],
                'total_assets': ['total assets'],
                'total_equity': ['total equity', "total stockholders' equity"],
                'eps': ['earnings per share (diluted)', 'eps (diluted)'],
            }
            
            for idx, row in df.iterrows():
                metric_name = str(row[0]).strip().lower() if pd.notna(row[0]) else ""
                
                for key, patterns in key_metrics.items():
                    if metric_name in patterns:
                        metrics[key] = {}
                        for col, fy in fiscal_years.items():
                            val = row[col]
                            if pd.notna(val) and str(val) != '-':
                                try:
                                    metrics[key][fy] = float(val)
                                except:
                                    pass
                        print(f"    Found {key}: Row {idx}")
                        break
            
            # Create one document per fiscal year
            for col, fiscal_year in fiscal_years.items():
                if fiscal_year == "FYNAN" or "TTM" in str(fiscal_year):
                    continue
                
                revenue = metrics.get('revenue', {}).get(fiscal_year)
                net_income = metrics.get('net_income', {}).get(fiscal_year)
                ebitda = metrics.get('ebitda', {}).get(fiscal_year)
                gross_profit = metrics.get('gross_profit', {}).get(fiscal_year)
                operating_income = metrics.get('operating_income', {}).get(fiscal_year)
                total_assets = metrics.get('total_assets', {}).get(fiscal_year)
                total_equity = metrics.get('total_equity', {}).get(fiscal_year)
                eps = metrics.get('eps', {}).get(fiscal_year)
                
                if revenue is None and net_income is None:
                    continue
                
                # Build structured text
                text_lines = [
                    "Entity: Oracle Financial Services Software Ltd",
                    "Statement: Income Statement / Financial Summary",
                    f"Fiscal Year: {fiscal_year}",
                    "Currency: INR",
                    "Unit: millions (except per share data)",
                    "",
                ]
                
                if revenue is not None:
                    text_lines.append(f"Revenue: INR {revenue:,.2f} million")
                if net_income is not None:
                    text_lines.append(f"Net Income: INR {net_income:,.2f} million")
                if ebitda is not None:
                    text_lines.append(f"EBITDA: INR {ebitda:,.2f} million")
                if gross_profit is not None:
                    text_lines.append(f"Gross Profit: INR {gross_profit:,.2f} million")
                if operating_income is not None:
                    text_lines.append(f"Operating Income: INR {operating_income:,.2f} million")
                if total_assets is not None:
                    text_lines.append(f"Total Assets: INR {total_assets:,.2f} million")
                if total_equity is not None:
                    text_lines.append(f"Total Equity: INR {total_equity:,.2f} million")
                if eps is not None:
                    text_lines.append(f"EPS (Diluted): INR {eps:,.2f}")
                
                text_lines.extend(["", f"Source: {blob_name}"])
                
                print(f"    Ingested: {fiscal_year} | Revenue={revenue} | NetIncome={net_income}")
                
                documents.append({
                    "text": "\n".join(text_lines),
                    "metadata": {
                        "company": "Oracle Financial Services Software Ltd",
                        "statement": "Income Statement",
                        "fiscal_year": fiscal_year,
                        "currency": "INR",
                        "unit": "millions",
                        "revenue": revenue,
                        "net_income": net_income,
                        "source": f"azure_blob/{blob_name}",
                        "document_type": "financial_row",
                        "file_type": "excel",
                        "container": get_config().azure_blob.container_name
                    }
                })
    
    except Exception as e:
        print(f"    [ERROR] Failed to parse Excel {blob_name}: {e}")
        import traceback
        traceback.print_exc()
    
    return documents


def parse_excel_fallback(df: pd.DataFrame, blob_name: str, sheet_name: str) -> List[Dict]:
    """
    Fallback Excel parser for non-transposed formats.
    Converts DataFrame to text rows.
    """
    documents = []
    
    try:
        # Convert DataFrame to text representation
        text_content = df.to_string(index=False)
        
        if len(text_content.strip()) < 50:
            return documents
        
        # Try to extract fiscal year from content
        fiscal_year = normalize_fiscal_year(text_content[:1000])
        
        structured_text = f"""Financial Data - Oracle Financial Services Software Ltd
Sheet: {sheet_name}
{f'Fiscal Year: {fiscal_year}' if fiscal_year else 'Fiscal Year: Unknown'}
Currency: INR
Unit: millions

{text_content}"""
        
        documents.append({
            "text": structured_text,
            "metadata": {
                "company": "Oracle Financial Services Software Ltd",
                "currency": "INR",
                "unit": "millions",
                "fiscal_year": fiscal_year or "Unknown",
                "source": f"azure_blob/{blob_name}",
                "document_type": "financial_data",
                "file_type": "excel",
                "sheet_name": sheet_name,
                "container": get_config().azure_blob.container_name
            }
        })
        
    except Exception as e:
        print(f"    [WARN] Fallback Excel parsing failed: {e}")
    
    return documents


def parse_pdf_with_context(content: bytes, blob_name: str) -> List[Dict]:
    """
    Parses PDF with financial context preservation.
    
    Handles:
    - Empty or None text from extract_text()
    - Pages with insufficient content
    - Parsing errors (logs warning, continues)
    """
    documents = []
    pdf_file = io.BytesIO(content)
    
    try:
        reader = PdfReader(pdf_file)
        total_pages = len(reader.pages)
        print(f"    PDF has {total_pages} pages")
        
        for page_num, page in enumerate(reader.pages, 1):
            try:
                page_text = page.extract_text()
                
                # Handle None or empty text
                if page_text is None:
                    print(f"    [WARN] Page {page_num}: extract_text() returned None, skipping")
                    continue
                
                page_text = page_text.strip()
                
                # Skip pages with insufficient content
                if len(page_text) < 50:
                    print(f"    [WARN] Page {page_num}: Text too short ({len(page_text)} chars), skipping")
                    continue
                
                fiscal_year = normalize_fiscal_year(page_text[:500])
                
                structured_text = f"""Financial Report - Oracle Financial Services Software Ltd
Currency: INR
Unit: millions
Page: {page_num}
{f'Fiscal Year: {fiscal_year}' if fiscal_year else ''}

{page_text}"""
                
                documents.append({
                    "text": structured_text,
                    "metadata": {
                        "company": "Oracle Financial Services Software Ltd",
                        "currency": "INR",
                        "unit": "millions",
                        "page": page_num,
                        "fiscal_year": fiscal_year or "Unknown",
                        "source": f"azure_blob/{blob_name}",
                        "document_type": "financial_report",
                        "file_type": "pdf",
                        "container": get_config().azure_blob.container_name
                    }
                })
                
                print(f"    [OK] Page {page_num}: {len(page_text)} chars extracted")
                
            except Exception as page_error:
                print(f"    [WARN] Page {page_num}: Failed to extract text: {page_error}, continuing...")
                continue
    
    except Exception as e:
        print(f"    [ERROR] Failed to parse PDF {blob_name}: {e}")
        import traceback
        traceback.print_exc()
    
    return documents


def load_azure_documents() -> List[Dict]:
    """
    Loads ALL PDF and Excel documents from Azure Blob Storage.
    
    Processes:
    - All .pdf files
    - All .xlsx files
    - All .xls files
    
    Returns:
        List of documents with text and metadata
    """
    print("=" * 60)
    print("LOADING AZURE BLOB DOCUMENTS")
    print("=" * 60)
    
    try:
        container_client = get_container_client()
        config = get_config().azure_blob
        print(f"Container: {config.container_name}")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Azure Blob: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    documents = []
    processed_blobs = []
    skipped_blobs = []
    error_blobs = []
    
    # Supported file extensions
    supported_extensions = ('.pdf', '.xlsx', '.xls', '.txt')
    
    try:
        # List ALL blobs in container
        all_blobs = list(container_client.list_blobs())
        print(f"\nFound {len(all_blobs)} total blobs in container")
        
        for blob in all_blobs:
            blob_name = blob.name
            blob_ext = blob_name.lower()
            
            # Check if file type is supported
            if not blob_ext.endswith(supported_extensions):
                skipped_blobs.append(blob_name)
                continue
            
            print(f"\n  [{len(processed_blobs) + 1}/{len(all_blobs)}] Processing: {blob_name}")
            print(f"    Size: {blob.size} bytes")
            
            try:
                # Download blob content
                content = download_blob(container_client, blob_name)
                print(f"    Downloaded: {len(content)} bytes")
                
                # Parse based on file type
                docs = []
                if blob_ext.endswith('.pdf'):
                    docs = parse_pdf_with_context(content, blob_name)
                elif blob_ext.endswith(('.xlsx', '.xls')):
                    docs = parse_excel_transposed(content, blob_name)
                elif blob_ext.endswith('.txt'):
                    # Handle TXT files from Azure Blob
                    try:
                        text_content = content.decode('utf-8')
                        if text_content.strip():
                            docs = [{
                                "text": text_content,
                                "metadata": {
                                    "source": f"azure_blob/{blob_name}",
                                    "document_type": "text",
                                    "file_type": "txt",
                                    "container": config.container_name
                                }
                            }]
                            print(f"    [OK] TXT file: {len(text_content)} chars")
                    except UnicodeDecodeError:
                        print(f"    [WARN] Failed to decode TXT as UTF-8, skipping")
                else:
                    print(f"    [WARN] Unsupported file type, skipping")
                    skipped_blobs.append(blob_name)
                    continue
                
                # Filter out empty documents
                valid_docs = [d for d in docs if d.get("text", "").strip()]
                
                if valid_docs:
                    documents.extend(valid_docs)
                    processed_blobs.append(blob_name)
                    total_text_len = sum(len(d.get("text", "")) for d in valid_docs)
                    print(f"    [OK] Extracted {len(valid_docs)} chunks ({total_text_len} total chars)")
                else:
                    print(f"    [WARN] No valid content extracted, skipping")
                    skipped_blobs.append(blob_name)
                
            except Exception as e:
                error_blobs.append((blob_name, str(e)))
                print(f"    [ERROR] Failed to process {blob_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
    except Exception as e:
        print(f"[ERROR] Failed to list blobs: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary statistics
    financial_rows = sum(1 for d in documents if d.get('metadata', {}).get('document_type') == 'financial_row')
    pdf_pages = sum(1 for d in documents if d.get('metadata', {}).get('document_type') == 'financial_report')
    
    print(f"\n{'=' * 60}")
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Processed blobs: {len(processed_blobs)}")
    print(f"Skipped blobs: {len(skipped_blobs)}")
    print(f"Error blobs: {len(error_blobs)}")
    print(f"Total chunks extracted: {len(documents)}")
    print(f"  - Financial rows: {financial_rows}")
    print(f"  - PDF pages: {pdf_pages}")
    print(f"  - Other: {len(documents) - financial_rows - pdf_pages}")
    
    if skipped_blobs:
        print(f"\nSkipped blobs (unsupported type or empty):")
        for blob in skipped_blobs[:10]:  # Show first 10
            print(f"  - {blob}")
        if len(skipped_blobs) > 10:
            print(f"  ... and {len(skipped_blobs) - 10} more")
    
    if error_blobs:
        print(f"\nBlobs with errors:")
        for blob, error in error_blobs[:5]:  # Show first 5
            print(f"  - {blob}: {error[:100]}")
        if len(error_blobs) > 5:
            print(f"  ... and {len(error_blobs) - 5} more errors")
    
    print("=" * 60)
    
    return documents
