# Processed Documents Storage

This folder contains processed PDFs and images that have been enhanced through the MCP preprocessing service.

## File Naming Convention

- **Processed PDFs:** `{original_name}_processed_{timestamp}.pdf`
- **Preprocessed Images:** `{original_name}_preprocessed_{timestamp}.{ext}`

## File Types

1. **Selective Preprocessing PDFs:** Mixed documents with enhanced pages and original high-quality pages
2. **Full Preprocessing PDFs:** Completely processed documents where all pages were enhanced
3. **Enhanced Images:** Single processed image files

## Timestamp Format

Files include timestamps in format: `YYYYMMDD_HHMMSS`
Example: `document_processed_20250923_152630.pdf`

## Cleanup

Unlike temporary files, these processed documents are permanently stored here for:
- Quality verification
- Comparison with originals
- Re-download by users
- Audit and compliance

## Retention Policy

Manual cleanup is required. Consider implementing automated cleanup for files older than:
- 30 days for regular processing
- 90 days for compliance documents