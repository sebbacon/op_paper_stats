#!/usr/bin/env python3
"""
Download papers that are in Google Scholar but not in PubMed.
Uses the same approach as paper_comparison.py to identify unique papers.
"""

import pandas as pd
import requests
import time
import csv
from pathlib import Path
from urllib.parse import urlparse
import hashlib
import logging
import asyncio
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import threading
import argparse
import json
import subprocess
import re
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_doi(doi):
    """Normalize DOI by removing prefixes and converting to lowercase."""
    if pd.isna(doi) or not doi:
        return ""

    doi = str(doi).strip().lower()

    prefixes = [
        "doi:",
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ]

    for prefix in prefixes:
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
            break

    return doi.strip()


def normalize_title(title):
    """Normalize title for matching by handling punctuation and whitespace variations."""
    if pd.isna(title) or not title:
        return ""

    import re

    title = str(title).strip().lower()
    title = re.sub(r"[.!?]+$", "", title)
    title = re.sub(r"[^\w\s\']", " ", title)
    title = re.sub(r"\s+", " ", title)

    words = title.strip().split()
    title = " ".join(words[:18])

    return title


def create_match_key(doi, title):
    """Create a matching key using DOI if available, otherwise normalized title."""
    doi_norm = normalize_doi(doi)
    if doi_norm:
        return f"doi:{doi_norm}"
    else:
        title_norm = normalize_title(title)
        return f"title:{title_norm}" if title_norm else ""


def find_gs_only_papers():
    """Find papers in Google Scholar but not in PubMed."""
    logger.info("Loading PubMed dataset...")
    papers_df = pd.read_csv("papers.csv")

    logger.info("Loading Google Scholar dataset...")
    op_papers_df = pd.read_csv("op_papers.csv")

    # Remove citations from Google Scholar data
    op_papers_df = op_papers_df[op_papers_df["Type"] != "CITATION"]

    # Fill missing values
    papers_df = papers_df.fillna("")
    op_papers_df = op_papers_df.fillna("")

    # Create match keys
    papers_df["match_key"] = papers_df.apply(
        lambda row: create_match_key(row["DOI"], row["Title"]), axis=1
    )
    op_papers_df["match_key"] = op_papers_df.apply(
        lambda row: create_match_key(row["DOI"], row["Title"]), axis=1
    )

    # Remove rows with empty match keys
    papers_df = papers_df[papers_df["match_key"] != ""]
    op_papers_df = op_papers_df[op_papers_df["match_key"] != ""]

    # De-duplicate within each dataset
    papers_df = papers_df.drop_duplicates(subset=["match_key"], keep="first")
    op_papers_df = op_papers_df.drop_duplicates(subset=["match_key"], keep="first")

    logger.info(f"PubMed papers: {len(papers_df)}")
    logger.info(f"Google Scholar papers: {len(op_papers_df)}")

    # Find papers in Google Scholar but not in PubMed
    pubmed_keys = set(papers_df["match_key"])
    gs_only = op_papers_df[~op_papers_df["match_key"].isin(pubmed_keys)]

    logger.info(f"Papers in Google Scholar but not PubMed: {len(gs_only)}")

    return gs_only


def create_filename_from_url(url, title):
    """Create a safe filename from URL and title."""
    # Get file extension from URL
    parsed = urlparse(url)
    path = parsed.path
    ext = Path(path).suffix if path else ".html"

    # Create safe filename from title
    if title:
        safe_title = "".join(
            c for c in title if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        safe_title = safe_title[:100]  # Limit length
        safe_title = safe_title.replace(" ", "_")
    else:
        # Fallback to URL hash
        safe_title = hashlib.md5(url.encode()).hexdigest()[:16]

    return f"{safe_title}{ext}"


async def download_article_async(session, url, filepath, timeout=30):
    """Download article content from URL asynchronously."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        async with session.get(url, headers=headers, timeout=timeout_obj) as response:
            status_code = response.status

            if response.status >= 400:
                error_msg = f"HTTP {response.status}: {response.reason}"
                return False, status_code, error_msg

            content = await response.read()

            async with aiofiles.open(filepath, "wb") as f:
                await f.write(content)

            return True, status_code, ""

    except asyncio.TimeoutError:
        error_msg = "Request timeout"
        logger.warning(f"Failed to download {url}: {error_msg}")
        return False, 0, error_msg

    except aiohttp.ClientError as e:
        error_msg = f"Client error: {e}"
        logger.warning(f"Failed to download {url}: {error_msg}")
        return False, 0, error_msg

    except Exception as e:
        error_msg = f"Request error: {e}"
        logger.warning(f"Failed to download {url}: {error_msg}")
        return False, 0, error_msg


def load_existing_downloads(csv_filename):
    """Load existing download records to make script idempotent."""
    if not Path(csv_filename).exists():
        return set()

    try:
        df = pd.read_csv(csv_filename)
        return set(df["article_url"].dropna())
    except Exception as e:
        logger.warning(f"Could not load existing downloads: {e}")
        return set()


def write_csv_header(csv_filename):
    """Write CSV header if file doesn't exist."""
    if not Path(csv_filename).exists():
        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "title",
                "authors",
                "year",
                "doi",
                "article_url",
                "filename",
                "filepath",
                "download_success",
                "http_status_code",
                "error_message",
                "ai_document_type",
                "ai_completeness",
                "ai_confidence",
                "has_openprescribing",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()


class ThreadSafeCSVWriter:
    """Thread-safe CSV writer for concurrent operations."""

    def __init__(self, csv_filename):
        self.csv_filename = csv_filename
        self.lock = threading.Lock()
        self.fieldnames = [
            "title",
            "authors",
            "year",
            "doi",
            "article_url",
            "filename",
            "filepath",
            "download_success",
            "http_status_code",
            "error_message",
            "ai_document_type",
            "ai_completeness",
            "ai_confidence",
            "has_openprescribing",
        ]

    def append_record(self, record):
        """Append a single record to CSV file in a thread-safe manner."""
        with self.lock:
            with open(self.csv_filename, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writerow(record)


def append_to_csv(csv_filename, record):
    """Append a single record to CSV file."""
    with open(csv_filename, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "title",
            "authors",
            "year",
            "doi",
            "article_url",
            "filename",
            "filepath",
            "download_success",
            "http_status_code",
            "error_message",
            "ai_document_type",
            "ai_completeness",
            "ai_confidence",
            "has_openprescribing",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(record)


def update_csv_with_classifications(csv_filename, ai_classifications):
    """Update existing CSV file with AI classification results."""
    if not Path(csv_filename).exists():
        logger.warning(f"CSV file not found: {csv_filename}")
        return
    
    try:
        # Read existing CSV
        df = pd.read_csv(csv_filename)
        
        # Add classification columns if they don't exist
        if "ai_document_type" not in df.columns:
            df["ai_document_type"] = ""
        if "ai_completeness" not in df.columns:
            df["ai_completeness"] = ""
        if "ai_confidence" not in df.columns:
            df["ai_confidence"] = ""
        
        # Update records with AI classifications
        updates_made = 0
        for index, row in df.iterrows():
            filepath = row.get("filepath", "")
            if filepath and str(filepath) in ai_classifications:
                classification = ai_classifications[str(filepath)]
                
                # Only update if classification fields are empty or unknown
                if (pd.isna(df.at[index, "ai_document_type"]) or 
                    df.at[index, "ai_document_type"] == "" or 
                    df.at[index, "ai_document_type"] == "unknown"):
                    
                    df.at[index, "ai_document_type"] = classification.get("document_type", "")
                    df.at[index, "ai_completeness"] = classification.get("completeness", "")
                    df.at[index, "ai_confidence"] = classification.get("confidence", "")
                    updates_made += 1
        
        # Write back to CSV
        df.to_csv(csv_filename, index=False)
        logger.info(f"Updated {updates_made} records in {csv_filename} with AI classifications")
        
    except Exception as e:
        logger.error(f"Error updating CSV with classifications: {e}")


def update_csv_with_openprescribing(csv_filename, openprescribing_data):
    """Update existing CSV file with OpenPrescribing detection results."""
    if not Path(csv_filename).exists():
        logger.warning(f"CSV file not found: {csv_filename}")
        return
    
    try:
        # Read existing CSV
        df = pd.read_csv(csv_filename)
        
        # Add has_openprescribing column if it doesn't exist
        if "has_openprescribing" not in df.columns:
            df["has_openprescribing"] = ""
        
        # Update records with OpenPrescribing detection results
        updates_made = 0
        for index, row in df.iterrows():
            filepath = row.get("filepath", "")
            if filepath and str(filepath) in openprescribing_data:
                has_op = openprescribing_data[str(filepath)]
                
                # Only update if field is empty
                if (pd.isna(df.at[index, "has_openprescribing"]) or 
                    df.at[index, "has_openprescribing"] == ""):
                    
                    df.at[index, "has_openprescribing"] = has_op
                    updates_made += 1
        
        # Write back to CSV
        df.to_csv(csv_filename, index=False)
        logger.info(f"Updated {updates_made} records in {csv_filename} with OpenPrescribing detection")
        
    except Exception as e:
        logger.error(f"Error updating CSV with OpenPrescribing detection: {e}")


def analyze_openprescribing_mentions(csv_filename):
    """
    Analyze the proportion of documents that contain "openprescribing" by document type and completeness.
    Returns a detailed breakdown of the proportions using data already stored in CSV.
    """
    if not Path(csv_filename).exists():
        logger.error(f"CSV file not found: {csv_filename}")
        return None

    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return None

    if df.empty:
        logger.info("CSV file is empty")
        return None

    # Filter for successfully downloaded files with classification data
    classified_df = df[
        (df["download_success"] == True) &
        (df["ai_document_type"].notna()) &
        (df["ai_document_type"] != "") &
        (df["ai_document_type"] != "unknown") &
        (df["ai_completeness"].notna()) &
        (df["ai_completeness"] != "") &
        (df["ai_completeness"] != "unknown")
    ]

    if classified_df.empty:
        logger.warning("No classified documents found in the dataset")
        return None

    # Count openprescribing mentions using CSV data
    openprescribing_counts = {
        'by_document_type': {},
        'by_completeness': {},
        'by_combination': {},
        'total_with_op': 0,
        'total_documents': len(classified_df)
    }

    for _, row in classified_df.iterrows():
        doc_type = row.get("ai_document_type", "")
        completeness = row.get("ai_completeness", "")
        has_op_str = row.get("has_openprescribing", "")
        
        # Convert string to boolean (CSV stores as string)
        has_openprescribing = str(has_op_str).lower() in ['true', '1', 'yes']

        # Initialize counters if needed
        if doc_type not in openprescribing_counts['by_document_type']:
            openprescribing_counts['by_document_type'][doc_type] = {'total': 0, 'with_op': 0}
        
        if completeness not in openprescribing_counts['by_completeness']:
            openprescribing_counts['by_completeness'][completeness] = {'total': 0, 'with_op': 0}
        
        combo_key = f"{doc_type}_{completeness}"
        if combo_key not in openprescribing_counts['by_combination']:
            openprescribing_counts['by_combination'][combo_key] = {
                'total': 0, 'with_op': 0, 'doc_type': doc_type, 'completeness': completeness
            }

        # Update counters
        openprescribing_counts['by_document_type'][doc_type]['total'] += 1
        openprescribing_counts['by_completeness'][completeness]['total'] += 1
        openprescribing_counts['by_combination'][combo_key]['total'] += 1

        if has_openprescribing:
            openprescribing_counts['by_document_type'][doc_type]['with_op'] += 1
            openprescribing_counts['by_completeness'][completeness]['with_op'] += 1
            openprescribing_counts['by_combination'][combo_key]['with_op'] += 1
            openprescribing_counts['total_with_op'] += 1

    return openprescribing_counts


def format_percentage(with_op, total):
    """Format percentage with count and total."""
    if total == 0:
        return "0.0% (0/0)"
    percentage = (with_op / total) * 100
    return f"{percentage:.1f}% ({with_op}/{total})"


def print_openprescribing_analysis(counts):
    """Print a formatted report of OpenPrescribing mentions."""
    if not counts:
        return

    print(f"\n## OpenPrescribing Mentions Analysis")
    
    total_with_op = counts['total_with_op']
    total_docs = counts['total_documents']
    overall_percentage = format_percentage(total_with_op, total_docs)
    
    print(f"\n### Overall Summary")
    print(f"- **Total classified documents:** {total_docs:,}")
    print(f"- **Documents mentioning OpenPrescribing:** {overall_percentage}")

    # By document type
    print(f"\n### By Document Type")
    doc_type_data = counts['by_document_type']
    
    # Sort by proportion (highest first)
    sorted_doc_types = sorted(
        doc_type_data.items(), 
        key=lambda x: x[1]['with_op'] / max(x[1]['total'], 1), 
        reverse=True
    )
    
    for doc_type, data in sorted_doc_types:
        display_name = doc_type.replace("_", " ").title()
        percentage = format_percentage(data['with_op'], data['total'])
        print(f"- **{display_name}:** {percentage}")

    # By completeness
    print(f"\n### By Document Completeness")
    completeness_data = counts['by_completeness']
    
    # Sort by proportion (highest first)
    sorted_completeness = sorted(
        completeness_data.items(), 
        key=lambda x: x[1]['with_op'] / max(x[1]['total'], 1), 
        reverse=True
    )
    
    for completeness, data in sorted_completeness:
        display_name = completeness.replace("_", " ").title()
        percentage = format_percentage(data['with_op'], data['total'])
        print(f"- **{display_name}:** {percentage}")

    # Detailed breakdown by combination (top 10 only)
    print(f"\n### Detailed Breakdown (Document Type × Completeness, Top 10)")
    combo_data = counts['by_combination']
    
    # Sort by proportion (highest first) and filter to existing combinations
    existing_combos = [(k, v) for k, v in combo_data.items() if v['total'] > 0]
    sorted_combos = sorted(
        existing_combos, 
        key=lambda x: x[1]['with_op'] / max(x[1]['total'], 1), 
        reverse=True
    )
    
    for combo_key, data in sorted_combos[:10]:  # Top 10 only
        doc_type_display = data['doc_type'].replace("_", " ").title()
        completeness_display = data['completeness'].replace("_", " ").title()
        percentage = format_percentage(data['with_op'], data['total'])
        print(f"- **{doc_type_display} + {completeness_display}:** {percentage}")


async def download_paper_task(session, csv_writer, data_dir, row, timeout=30):
    """Download a single paper asynchronously."""
    article_url = row.get("ArticleURL", "")
    title = row.get("Title", "")

    if not article_url or pd.isna(article_url):
        return False, "no_url"

    # Create filename
    filename = create_filename_from_url(article_url, title)
    filepath = data_dir / filename

    logger.info(f"Downloading: {title[:50]}... -> {filename}")

    # Download the article
    success, status_code, error_msg = await download_article_async(
        session, article_url, filepath, timeout
    )

    # Record the attempt
    record = {
        "title": title,
        "authors": row.get("Authors", ""),
        "year": row.get("Year", ""),
        "doi": row.get("DOI", ""),
        "article_url": article_url,
        "filename": filename if success else "",
        "filepath": str(filepath) if success else "",
        "download_success": success,
        "http_status_code": status_code,
        "error_message": error_msg,
        "ai_document_type": "",
        "ai_completeness": "",
        "ai_confidence": "",
        "has_openprescribing": "",
    }

    # Write to CSV immediately (thread-safe)
    csv_writer.append_record(record)

    return success, "downloaded" if success else "failed"


async def download_papers_concurrently(
    papers_to_download,
    csv_filename,
    data_dir="data",
    max_concurrent=10,
    timeout=30,
    batch_size=50,
):
    """Download papers concurrently with rate limiting."""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    csv_writer = ThreadSafeCSVWriter(csv_filename)

    # Configure aiohttp session with connection limits
    connector = aiohttp.TCPConnector(
        limit=max_concurrent,
        limit_per_host=min(
            5, max_concurrent
        ),  # Max 5 concurrent connections per host, or fewer if max_concurrent is low
        ttl_dns_cache=300,
        use_dns_cache=True,
    )

    timeout_obj = aiohttp.ClientTimeout(total=timeout, connect=min(10, timeout // 3))

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout_obj
    ) as session:
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_semaphore(row):
            async with semaphore:
                # Add small delay to be respectful to servers
                await asyncio.sleep(0.1)
                return await download_paper_task(
                    session, csv_writer, data_dir, row, timeout
                )

        # Create tasks for all papers
        tasks = [
            download_with_semaphore(row) for _, row in papers_to_download.iterrows()
        ]

        # Execute with progress tracking
        downloaded_count = 0
        failed_count = 0

        # Process in batches to avoid overwhelming the system
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1} ({len(batch)} papers)"
            )

            try:
                results = await asyncio.gather(*batch, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Task failed with exception: {result}")
                        failed_count += 1
                    else:
                        success, status = result
                        if success:
                            downloaded_count += 1
                        else:
                            failed_count += 1

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                failed_count += len(batch)

            # Brief pause between batches
            await asyncio.sleep(1)

    return downloaded_count, failed_count


def stats_papers(args):
    """Print summary statistics from the CSV file."""
    # Handle sample classification mode
    if hasattr(args, "sample") and args.sample:
        data_dir = getattr(args, "data_dir", "data")
        sample_classify_documents(data_dir, args.sample)
        return

    csv_filename = args.csv_file

    if not Path(csv_filename).exists():
        logger.error(f"CSV file not found: {csv_filename}")
        return

    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return

    if df.empty:
        logger.info("CSV file is empty")
        return

    print(f"\n# Download Statistics for {csv_filename}")

    # Overall stats
    total_papers = len(df)
    successful = len(df[df["download_success"] == True])
    failed = len(df[df["download_success"] == False])
    success_rate = (successful / total_papers * 100) if total_papers > 0 else 0

    print(f"\n## Overall Summary")
    print(f"- **Total papers processed:** {total_papers:,}")
    print(f"- **Successful downloads:** {successful:,} ({success_rate:.1f}%)")
    print(f"- **Failed downloads:** {failed:,} ({100-success_rate:.1f}%)")

    # Status code breakdown
    if "http_status_code" in df.columns:
        print(f"\n## HTTP Status Code Breakdown")
        status_counts = df["http_status_code"].value_counts().sort_index()

        for status_code, count in status_counts.items():
            percentage = count / total_papers * 100
            status_desc = get_status_description(status_code)
            print(f"- **{status_code} {status_desc}:** {count:,} ({percentage:.1f}%)")

    # Error analysis
    if "error_message" in df.columns and not df["error_message"].isna().all():
        print(f"\n## Error Analysis")
        error_df = df[df["error_message"].notna() & (df["error_message"] != "")]

        if not error_df.empty:
            # Group similar errors
            error_patterns = {}
            for error_msg in error_df["error_message"]:
                # Extract error type (first part before colon or first few words)
                if ":" in str(error_msg):
                    error_type = str(error_msg).split(":")[0]
                else:
                    error_type = " ".join(str(error_msg).split()[:3])

                error_patterns[error_type] = error_patterns.get(error_type, 0) + 1

            for error_type, count in sorted(
                error_patterns.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                percentage = count / total_papers * 100
                print(f"- **{error_type}:** {count:,} ({percentage:.1f}%)")

    # File analysis
    print(f"\n## File Analysis")
    data_dir = getattr(args, "data_dir", "data")
    report_file = f"{data_dir}_analysis.json"
    regenerate = getattr(args, "regenerate", False)
    classify_with_ai = getattr(args, "classify", False)

    file_analysis = analyze_downloaded_files(
        data_dir, df, report_file, regenerate, classify_with_ai, csv_filename
    )

    if file_analysis["total_files"] > 0:
        print(f"- **Files found on disk:** {file_analysis['total_files']:,}")
        print(f"- **Total size:** {format_bytes(file_analysis['total_size'])}")
        print(f"- **Average file size:** {format_bytes(file_analysis['avg_size'])}")
        print(f"- **Size range:** {format_bytes(file_analysis['min_size'])} - {format_bytes(file_analysis['max_size'])}")

        # Word count statistics
        if file_analysis.get("total_words", 0) > 0:
            print(f"- **Total words:** {file_analysis['total_words']:,}")
            print(f"- **Average words per file:** {file_analysis['avg_words']:,.0f}")

        # OpenPrescribing mentions
        openprescribing_count = file_analysis.get("openprescribing_count", 0)
        if openprescribing_count > 0:
            percentage = openprescribing_count / file_analysis["total_files"] * 100
            print(f"- **Mention OpenPrescribing:** {openprescribing_count:,} files ({percentage:.1f}%)")

        # File type breakdown
        if file_analysis["file_types"]:
            print(f"\n### File Type Distribution")
            for file_type, type_info in sorted(
                file_analysis["file_types"].items(),
                key=lambda x: x[1]["count"],
                reverse=True,
            ):
                count = type_info["count"]
                size = type_info["total_size"]
                avg_size = type_info["avg_size"]
                words = type_info.get("total_words", 0)
                avg_words = type_info.get("avg_words", 0)
                op_count = type_info.get("openprescribing_count", 0)
                percentage = count / file_analysis["total_files"] * 100

                word_info = (
                    f", {words:,} words ({avg_words:,.0f} avg)" if words > 0 else ""
                )
                op_info = f", {op_count} w/OP" if op_count > 0 else ""
                print(
                    f"- **{file_type.upper()}:** {count:,} files ({percentage:.1f}%) - "
                    f"{format_bytes(size)} total, {format_bytes(avg_size)} avg{word_info}{op_info}"
                )

        # AI Classification results
        if file_analysis.get("ai_classification_done", False):
            document_types = file_analysis.get("document_types", {})
            completeness_types = file_analysis.get("completeness_types", {})

            if document_types and any(k != "unknown" for k in document_types.keys()):
                print(f"\n### AI Document Type Classification")
                for doc_type, info in sorted(
                    document_types.items(), key=lambda x: x[1]["count"], reverse=True
                ):
                    if doc_type != "unknown" or info["count"] > 0:
                        count = info["count"]
                        words = info["total_words"]
                        op_count = info.get("openprescribing_count", 0)
                        percentage = count / file_analysis["total_files"] * 100
                        avg_words = words / count if count > 0 else 0

                        display_name = doc_type.replace("_", " ").title()
                        op_info = f", {op_count} w/OP" if op_count > 0 else ""
                        print(
                            f"- **{display_name}:** {count:,} files ({percentage:.1f}%) - {words:,} words ({avg_words:,.0f} avg){op_info}"
                        )

            if completeness_types and any(
                k != "unknown" for k in completeness_types.keys()
            ):
                print(f"\n### Document Completeness Classification")
                for completeness, info in sorted(
                    completeness_types.items(),
                    key=lambda x: x[1]["count"],
                    reverse=True,
                ):
                    if completeness != "unknown" or info["count"] > 0:
                        count = info["count"]
                        words = info["total_words"]
                        op_count = info.get("openprescribing_count", 0)
                        percentage = count / file_analysis["total_files"] * 100
                        avg_words = words / count if count > 0 else 0

                        display_name = completeness.replace("_", " ").title()
                        op_info = f", {op_count} w/OP" if op_count > 0 else ""
                        print(
                            f"- **{display_name}:** {count:,} files ({percentage:.1f}%) - {words:,} words ({avg_words:,.0f} avg){op_info}"
                        )
        elif classify_with_ai:
            print(f"\n### AI Classification")
            print("- **Status:** In progress...")

        # Show analysis timestamp
        import datetime

        timestamp = file_analysis.get("analysis_timestamp", time.time())
        date_str = datetime.datetime.fromtimestamp(timestamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        print(f"\n### Analysis Metadata")
        print(f"- **Analysis completed:** {date_str}")
        if file_analysis.get("ai_classification_done", False):
            print(f"- **AI classification:** Completed")
        if not regenerate and Path(report_file).exists():
            print(f"- **Report cached in:** {report_file}")
    else:
        print(f"- **Files found:** None")

    # Year distribution (if available)
    if "year" in df.columns and not df["year"].isna().all():
        print(f"\n## Year Distribution (Top 10)")
        year_counts = df["year"].value_counts().head(10)
        for year, count in year_counts.items():
            if pd.notna(year):
                percentage = count / total_papers * 100
                print(f"- **{year}:** {count:,} papers ({percentage:.1f}%)")

    # Individual AI classifications (if requested)
    if getattr(args, 'show_classifications', False) and "ai_document_type" in df.columns:
        classified_df = df[
            (df["ai_document_type"].notna()) & 
            (df["ai_document_type"] != "") & 
            (df["ai_document_type"] != "unknown")
        ]
        
        if not classified_df.empty:
            print(f"\n## Individual AI Classifications")
            print(f"*Showing {min(10, len(classified_df))} most recent*")
            
            # Sort by index (most recent first, assuming CSV is in chronological order)
            display_df = classified_df.tail(10)[::-1]  # Last 10, reversed
            
            for _, row in display_df.iterrows():
                title = row.get("title", "Unknown")[:60]
                doc_type = str(row.get("ai_document_type", "")).replace("_", " ").title()
                completeness = str(row.get("ai_completeness", "")).replace("_", " ").title()
                confidence = row.get("ai_confidence", "")
                
                # Format confidence
                try:
                    confidence_val = float(confidence)
                    confidence_str = f"{confidence_val:.2f}"
                except (ValueError, TypeError):
                    confidence_str = "N/A"
                
                print(f"\n### {title}...")
                print(f"- **Type:** {doc_type}")
                print(f"- **Completeness:** {completeness}")
                print(f"- **Confidence:** {confidence_str}")

    # Recent activity
    if args.recent and "filepath" in df.columns:
        print(f"\n## Recent Downloads")
        recent_files = []
        for _, row in df[df["download_success"] == True].iterrows():
            filepath = row.get("filepath", "")
            if filepath and Path(filepath).exists():
                mtime = Path(filepath).stat().st_mtime
                recent_files.append((mtime, row))

        recent_files.sort(reverse=True)
        for mtime, row in recent_files[:5]:
            import datetime

            date_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            title = row.get("title", "Unknown")[:50]
            print(f"- **{date_str}:** {title}...")

    # OpenPrescribing analysis (if classification data is available)
    if "ai_document_type" in df.columns and "ai_completeness" in df.columns:
        openprescribing_counts = analyze_openprescribing_mentions(csv_filename)
        if openprescribing_counts:
            print_openprescribing_analysis(openprescribing_counts)

    print()


def get_status_description(status_code):
    """Get human-readable description for HTTP status codes."""
    status_descriptions = {
        0: "No Response",
        200: "OK",
        201: "Created",
        202: "Accepted",
        204: "No Content",
        301: "Moved Permanently",
        302: "Found",
        304: "Not Modified",
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        408: "Request Timeout",
        429: "Too Many Requests",
        500: "Internal Server Error",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout",
    }
    return status_descriptions.get(int(status_code), "Unknown")


def format_bytes(bytes_val):
    """Format bytes into human readable format."""
    if bytes_val == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while bytes_val >= 1024 and i < len(units) - 1:
        bytes_val /= 1024
        i += 1

    return f"{bytes_val:.1f} {units[i]}"


def extract_full_text(file_path, file_type):
    """Extract full text from file without character limits."""
    return extract_text_for_classification(file_path, file_type, max_chars=None)


def extract_text_for_classification(file_path, file_type, max_chars=8000):
    """Extract text from file for AI classification, limited to max_chars."""
    try:
        if file_type == "pdf":
            # Use pdftotext if available
            try:
                result = subprocess.run(
                    ["pdftotext", str(file_path), "-"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    text = result.stdout
                else:
                    return ""
            except (
                subprocess.TimeoutExpired,
                subprocess.SubprocessError,
                FileNotFoundError,
            ):
                return ""

        elif file_type in ["html", "xml"]:
            # Extract text from HTML/XML using BeautifulSoup
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Use BeautifulSoup to properly parse HTML and extract text
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(content, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text content
                text = soup.get_text()
            except ImportError:
                # Fallback to regex if BeautifulSoup not available
                text = re.sub(r"<[^>]+>", " ", content)
                text = re.sub(r"&[a-zA-Z]+;", " ", text)
            except:
                return ""

        elif file_type in ["text", "csv", "json"]:
            # Plain text files
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except:
                return ""

        elif file_type in ["docx", "office"]:
            # Try to extract text from Office documents using unzip
            try:
                result = subprocess.run(
                    ["unzip", "-p", str(file_path), "word/document.xml"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    xml_content = result.stdout
                    # Remove XML tags
                    text = re.sub(r"<[^>]+>", " ", xml_content)
                else:
                    return ""
            except (
                subprocess.TimeoutExpired,
                subprocess.SubprocessError,
                FileNotFoundError,
            ):
                return ""
        else:
            # For other file types, don't extract text
            return ""

        # Clean up whitespace and limit length
        text = " ".join(text.split())  # Normalize whitespace
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars]
        return text

    except Exception:
        return ""


def create_batch_request(documents_batch):
    """Create a batch request for multiple documents."""

    # Create batch request with multiple documents
    requests_data = []

    for i, (file_path, text) in enumerate(documents_batch):
        prompt = f"""Please analyze this document and classify it into two dimensions:

1. Document Type (choose one):
   - full_academic_paper: An academic paper, in a proper journal, likely to have been peer reviewed, not a preprint, not an appendix or supplementary material
   - editorial: Editorial or opinion piece
   - briefing: Policy brief, technical brief, or summary document
   - poster: Conference poster or poster abstract
   - conference_talk: Conference presentation or talk abstract
   - preprint: Preprint or working paper
   - thesis: academic thesis
   - academic_supplementary: appendix or other supplementary material to a paper
   - academic_poster: an academic poster for a conference
   - academic_other: Other academic content (review, commentary, lecture notes, oral sessions, etc.)
   - non_academic_other: Non-academic content

2. Completeness (choose one):
   - full_document: Appears to be complete document, not only the abstract. Complete documents tend to have a substantial introduction or other narrative text following the abstract's conclusion.
   - abstract_only: Appears to be just an abstract or very brief summary

Please respond with ONLY a JSON object in this exact format:
{{"document_type": "your_classification", "completeness": "your_classification", "confidence": 0.95}}

The confidence should be a number between 0 and 1 indicating how certain you are.

Document text (first 8000 characters):
{text}"""

        requests_data.append(
            {
                "custom_id": f"doc_{i}",
                "params": {
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )

    return requests_data


def submit_batch_request(requests_data, api_key):
    """Submit a batch request to Claude API."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages/batches",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "message-batches-2024-09-24",
            },
            json={"requests": requests_data},
            timeout=30,
        )

        if response.status_code == 200:
            batch_data = response.json()
            return batch_data["id"]
        else:
            logger.error(
                f"Batch submission failed: {response.status_code} - {response.text}"
            )
            return None

    except Exception as e:
        logger.error(f"Error submitting batch request: {e}")
        return None


def get_batch_results(batch_id, api_key):
    """Poll for batch results."""

    try:
        response = requests.get(
            f"https://api.anthropic.com/v1/messages/batches/{batch_id}",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "message-batches-2024-09-24",
            },
            timeout=30,
        )

        if response.status_code == 200:
            batch_data = response.json()
            return batch_data
        else:
            logger.error(f"Batch status check failed: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"Error checking batch status: {e}")
        return None


def download_batch_results(results_url, api_key):
    """Download the batch results file."""

    try:
        response = requests.get(
            results_url,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout=60,
        )

        if response.status_code == 200:
            # Parse JSONL response
            results = []
            for line in response.text.strip().split("\n"):
                if line:
                    results.append(json.loads(line))
            return results
        else:
            logger.error(f"Batch results download failed: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"Error downloading batch results: {e}")
        return None


def classify_documents_batch(documents_to_classify):
    """Classify multiple documents using Claude batch API, with fallback to individual requests."""

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not found. Skipping AI classification.")
        return {}

    if not documents_to_classify:
        return {}

    # Try batch API first
    logger.info(
        f"Attempting batch classification for {len(documents_to_classify)} documents..."
    )

    # Create batch request
    requests_data = create_batch_request(documents_to_classify)

    # Submit batch
    batch_id = submit_batch_request(requests_data, api_key)

    if batch_id:
        logger.info(f"Batch submitted with ID: {batch_id}")
        logger.info("Waiting for batch processing to complete...")

        # Poll for completion
        max_wait_time = 600  # 10 minutes max
        poll_interval = 10  # Check every 10 seconds
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            batch_status = get_batch_results(batch_id, api_key)
            if not batch_status:
                logger.error("Failed to check batch status")
                break

            status = batch_status.get("processing_status")
            logger.info(f"Batch status: {status}")

            if status == "ended":
                # Download results
                results_url = batch_status.get("results_url")
                if not results_url:
                    logger.error("No results URL in completed batch")
                    break

                results = download_batch_results(results_url, api_key)
                if not results:
                    logger.error("Failed to download batch results")
                    break

                # Parse results and match back to file paths
                classifications = {}
                for i, result in enumerate(results):
                    if i < len(documents_to_classify):
                        file_path = documents_to_classify[i][0]

                        if result.get("result", {}).get("type") == "succeeded":
                            content = result["result"]["message"]["content"][0][
                                "text"
                            ].strip()

                            try:
                                classification = json.loads(content)
                                classifications[str(file_path)] = classification
                            except json.JSONDecodeError:
                                # Fallback: try to extract JSON from response
                                json_match = re.search(r"\{[^}]*\}", content)
                                if json_match:
                                    try:
                                        classification = json.loads(json_match.group())
                                        classifications[str(file_path)] = classification
                                    except:
                                        classifications[str(file_path)] = {
                                            "document_type": "unknown",
                                            "completeness": "unknown",
                                            "confidence": 0,
                                        }
                                else:
                                    classifications[str(file_path)] = {
                                        "document_type": "unknown",
                                        "completeness": "unknown",
                                        "confidence": 0,
                                    }
                        else:
                            classifications[str(file_path)] = {
                                "document_type": "unknown",
                                "completeness": "unknown",
                                "confidence": 0,
                            }

                logger.info(
                    f"Successfully classified {len(classifications)} documents using batch API"
                )
                return classifications

            elif status == "failed":
                logger.error("Batch processing failed")
                break

            # Wait before next poll
            time.sleep(poll_interval)
            elapsed_time += poll_interval

        if elapsed_time >= max_wait_time:
            logger.error("Batch processing timed out")

    # Fallback to individual requests
    logger.warning("Batch API failed, falling back to individual requests...")
    logger.info("This will be slower but should work reliably")

    classifications = {}
    for i, (file_path, text) in enumerate(documents_to_classify):
        logger.info(
            f"Classifying document {i+1}/{len(documents_to_classify)}: {file_path.name}"
        )

        classification = classify_document_with_claude_fallback(text, file_path)
        classifications[str(file_path)] = classification

        # Small delay to be respectful to API
        time.sleep(0.5)

    logger.info(
        f"Successfully classified {len(classifications)} documents using individual requests"
    )
    return classifications


def sample_classify_documents(data_dir, sample_size=10):
    """Classify a random sample of documents for verification."""
    import random

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("**Error:** ANTHROPIC_API_KEY not found. Please set your API key first.")
        print("```bash")
        print("export ANTHROPIC_API_KEY='your_api_key_here'")
        print("```")
        return

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"**Error:** Data directory not found: {data_dir}")
        return

    # Load the download CSV to get original URLs and download status
    csv_filename = "gs_only_downloads.csv"
    url_mapping = {}
    successfully_downloaded_files = set()

    if Path(csv_filename).exists():
        try:
            df = pd.read_csv(csv_filename)
            # Create mapping from filepath to original URL and track successful downloads
            for _, row in df.iterrows():
                filepath = row.get("filepath", "")
                article_url = row.get("article_url", "")
                download_success = row.get("download_success", False)

                # Skip rows with NaN or empty values
                if (
                    pd.isna(filepath)
                    or pd.isna(article_url)
                    or not filepath
                    or not article_url
                ):
                    continue

                # Convert to string and create mapping
                filepath_str = str(filepath).strip()
                article_url_str = str(article_url).strip()

                if filepath_str and article_url_str:
                    url_mapping[str(Path(filepath_str))] = article_url_str
                    
                    # Track successfully downloaded files
                    if download_success:
                        successfully_downloaded_files.add(str(Path(filepath_str)))
        except Exception as e:
            print(f"**Warning:** Could not load URL mapping from {csv_filename}: {e}")

    # Show filtering information
    if successfully_downloaded_files:
        print(f"**Found {len(successfully_downloaded_files)} successfully downloaded files**")
        print("Only these files will be considered for classification")

    # Find all files recursively
    all_files = []
    for file_path in data_path.rglob("*"):
        if file_path.is_file():
            all_files.append(file_path)

    if not all_files:
        print(f"**Error:** No files found in {data_dir}")
        return

    # Filter to classifiable file types and get file info
    classifiable_files = []
    for file_path in all_files:
        try:
            # Use file command to determine file type
            result = subprocess.run(
                ["file", "-b", str(file_path)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                file_output = result.stdout.strip().lower()
                file_type = classify_file_type(file_output)

                if file_type in ["pdf", "html", "text", "docx", "office"]:
                    # Only include files that were successfully downloaded or if we don't have download status info
                    if not successfully_downloaded_files or str(file_path) in successfully_downloaded_files:
                        # Get file size and word count for context
                        size = file_path.stat().st_size
                        word_count = count_words_in_file(file_path, file_type)

                        # Check for OpenPrescribing mention in full document
                        text = extract_full_text(file_path, file_type)
                        has_openprescribing = (
                            "openprescribing" in text.lower() if text else False
                        )

                        classifiable_files.append(
                            {
                                "path": file_path,
                                "file_type": file_type,
                                "size": size,
                                "word_count": word_count,
                                "has_openprescribing": has_openprescribing,
                            }
                        )
        except:
            continue

    if not classifiable_files:
        print(f"**Error:** No classifiable documents found in {data_dir}")
        return

    # Sample random files
    sample_count = min(sample_size, len(classifiable_files))
    sampled_files = random.sample(classifiable_files, sample_count)

    print(f"**Sampling {sample_count} documents from {len(classifiable_files)} classifiable files**")
    print(f"**Estimated cost:** ~${(sample_count * 1200 * 0.80 / 1000000):.3f}")
    print()

    # Prepare documents for classification
    documents_to_classify = []
    for file_info in sampled_files:
        text = extract_text_for_classification(
            file_info["path"], file_info["file_type"]
        )
        if text:
            documents_to_classify.append((file_info["path"], text))

    if not documents_to_classify:
        print("**Error:** Could not extract text from any sampled documents")
        return

    # Classify documents using batch API
    print(f"**Classifying {len(documents_to_classify)} documents...**")
    ai_classifications = classify_documents_batch(documents_to_classify)

    # Display results
    print("\n# Sample Classification Results")

    for i, file_info in enumerate(sampled_files):
        file_path = file_info["path"]
        classification = ai_classifications.get(
            str(file_path),
            {"document_type": "unknown", "completeness": "unknown", "confidence": 0},
        )

        print(f"\n## Document {i+1}/{len(sampled_files)}")
        print(f"- **File:** {file_path.name}")

        # Show original ArticleURL if available, otherwise local path
        original_url = url_mapping.get(str(file_path))
        if original_url:
            print(f"- **Original URL:** {original_url}")
        else:
            print(f"- **Path:** {file_path}")

        print(f"- **Type:** {file_info['file_type'].upper()}")
        print(f"- **Size:** {format_bytes(file_info['size'])}")
        print(f"- **Words:** {file_info['word_count']:,}")

        # Show OpenPrescribing mention
        has_op = file_info.get("has_openprescribing", False)
        op_indicator = "Yes" if has_op else "No"
        print(f"- **OpenPrescribing:** {op_indicator}")

        print(f"\n### AI Classification")
        doc_type = classification["document_type"].replace("_", " ").title()
        completeness = classification["completeness"].replace("_", " ").title()
        confidence = classification.get("confidence", 0)

        print(f"- **Document Type:** {doc_type}")
        print(f"- **Completeness:** {completeness}")
        print(f"- **Confidence:** {confidence:.2f}")

        # Show original URL if available, otherwise create clickable file link
        if original_url:
            print(f"- **Original:** {original_url}")
        else:
            try:
                # Convert to absolute path first
                abs_path = file_path.resolve()
                file_uri = abs_path.as_uri()
                print(f"- **Open:** {file_uri}")
            except Exception as e:
                # Fallback to just showing the path
                print(f"- **Path:** {file_path}")

        # Show first few lines of text for verification
        if file_info["word_count"] > 0:
            text_preview = extract_text_for_classification(
                file_path, file_info["file_type"], max_chars=300
            )
            if text_preview:
                preview = " ".join(text_preview.split()[:50])  # First 50 words
                if len(text_preview) > len(preview):
                    preview += "..."
                print(f"- **Preview:** {preview}")

        print("---")

    # Summary
    if ai_classifications:
        doc_types = {}
        completeness_types = {}
        confidences = []

        for classification in ai_classifications.values():
            doc_type = classification["document_type"]
            completeness = classification["completeness"]
            confidence = classification.get("confidence", 0)

            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            completeness_types[completeness] = (
                completeness_types.get(completeness, 0) + 1
            )
            if confidence > 0:
                confidences.append(confidence)

        print(f"\n## Sample Summary")
        print(f"- **Documents classified:** {len(ai_classifications)}")

        if doc_types:
            print(f"\n### Document Types")
            for doc_type, count in sorted(
                doc_types.items(), key=lambda x: x[1], reverse=True
            ):
                display_name = doc_type.replace("_", " ").title()
                print(f"- **{display_name}:** {count}")

        if completeness_types:
            print(f"\n### Completeness")
            for completeness, count in sorted(
                completeness_types.items(), key=lambda x: x[1], reverse=True
            ):
                display_name = completeness.replace("_", " ").title()
                print(f"- **{display_name}:** {count}")

        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            print(f"\n- **Average confidence:** {avg_confidence:.2f}")

    print(
        f"\n**Tip:** To classify all documents: `python download_gs_only_papers.py stats --classify`"
    )


def classify_document_with_claude_fallback(text, file_path):
    """Fallback single document classification (original method)."""

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not found. Skipping AI classification.")
        return {"document_type": "unknown", "completeness": "unknown", "confidence": 0}

    if not text.strip():
        return {"document_type": "unknown", "completeness": "unknown", "confidence": 0}

    prompt = f"""Please analyze this document and classify it into two dimensions:

1. Document Type (choose one):
   - full_academic_paper: Complete research paper with methods, results, discussion
   - editorial: Editorial or opinion piece
   - briefing: Policy brief, technical brief, or summary document
   - poster: Conference poster or poster abstract
   - conference_talk: Conference presentation or talk abstract
   - preprint: Preprint or working paper
   - academic_other: Other academic content (review, commentary, etc.)
   - non_academic_other: Non-academic content

2. Completeness (choose one):
   - full_document: Contains substantial content, appears to be complete document
   - abstract_only: Appears to be just an abstract or very brief summary

Please respond with ONLY a JSON object in this exact format:
{{"document_type": "your_classification", "completeness": "your_classification", "confidence": 0.95}}

The confidence should be a number between 0 and 1 indicating how certain you are.

Document text (first 8000 characters):
{text}"""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            content = result["content"][0]["text"].strip()

            # Try to parse JSON response
            try:
                classification = json.loads(content)
                return classification
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from response
                json_match = re.search(r"\{[^}]*\}", content)
                if json_match:
                    try:
                        classification = json.loads(json_match.group())
                        return classification
                    except:
                        pass

                logger.warning(
                    f"Could not parse Claude response for {file_path}: {content}"
                )
                return {
                    "document_type": "unknown",
                    "completeness": "unknown",
                    "confidence": 0,
                }
        else:
            logger.warning(f"Claude API error {response.status_code} for {file_path}")
            return {
                "document_type": "unknown",
                "completeness": "unknown",
                "confidence": 0,
            }

    except Exception as e:
        logger.warning(f"Error calling Claude API for {file_path}: {e}")
        return {"document_type": "unknown", "completeness": "unknown", "confidence": 0}


def count_words_in_file(file_path, file_type):
    """Count words in a file based on its type."""
    try:
        if file_type == "pdf":
            # Use pdftotext if available, otherwise try basic extraction
            try:
                result = subprocess.run(
                    ["pdftotext", str(file_path), "-"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    text = result.stdout
                else:
                    return 0
            except (
                subprocess.TimeoutExpired,
                subprocess.SubprocessError,
                FileNotFoundError,
            ):
                return 0

        elif file_type in ["html", "xml"]:
            # Extract text from HTML/XML
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                # Remove HTML tags and extract text
                text = re.sub(r"<[^>]+>", " ", content)
                text = re.sub(r"&[a-zA-Z]+;", " ", text)  # Remove HTML entities
            except:
                return 0

        elif file_type in ["text", "csv", "json"]:
            # Plain text files
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except:
                return 0

        elif file_type in ["docx", "office"]:
            # Try to extract text from Office documents using unzip + grep
            try:
                result = subprocess.run(
                    ["unzip", "-p", str(file_path), "word/document.xml"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    xml_content = result.stdout
                    # Remove XML tags
                    text = re.sub(r"<[^>]+>", " ", xml_content)
                else:
                    return 0
            except (
                subprocess.TimeoutExpired,
                subprocess.SubprocessError,
                FileNotFoundError,
            ):
                return 0
        else:
            # For other file types, don't count words
            return 0

        # Count words
        words = re.findall(r"\b\w+\b", text.lower())
        return len(words)

    except Exception:
        return 0


def analyze_downloaded_files(
    data_dir, df=None, report_file=None, regenerate=False, classify_with_ai=False, csv_filename=None
):
    """Analyze all files in the data directory and subdirectories."""

    # Check if we have a cached report
    if report_file and Path(report_file).exists() and not regenerate:
        logger.info(f"Loading cached analysis from {report_file}")
        try:
            with open(report_file, "r") as f:
                cached_data = json.load(f)
                # If AI classification wasn't done before but is requested now, regenerate
                if classify_with_ai and not cached_data.get(
                    "ai_classification_done", False
                ):
                    logger.info(
                        "AI classification requested but not in cache, regenerating..."
                    )
                else:
                    return cached_data
        except Exception as e:
            logger.warning(f"Could not load cached report: {e}, regenerating...")

    logger.info("Analyzing downloaded files...")
    if classify_with_ai:
        logger.info(
            "AI classification enabled - this will take longer and requires ANTHROPIC_API_KEY"
        )

    data_path = Path(data_dir)
    if not data_path.exists():
        return {
            "total_files": 0,
            "total_size": 0,
            "avg_size": 0,
            "min_size": 0,
            "max_size": 0,
            "total_words": 0,
            "avg_words": 0,
            "file_types": {},
            "document_types": {},
            "completeness_types": {},
            "ai_classification_done": classify_with_ai,
            "analysis_timestamp": time.time(),
        }

    # Load download status information if available
    successfully_downloaded_files = set()
    if df is not None:
        # Filter for successfully downloaded files
        successful_downloads = df[df["download_success"] == True]
        total_downloads = len(df)
        successful_count = len(successful_downloads)
        
        for _, row in successful_downloads.iterrows():
            filepath = row.get("filepath", "")
            if filepath and not pd.isna(filepath):
                successfully_downloaded_files.add(str(Path(filepath)))
        
        if classify_with_ai:
            logger.info(f"AI classification will only process {successful_count}/{total_downloads} successfully downloaded files")

    # Find all files recursively
    all_files = []
    for file_path in data_path.rglob("*"):
        if file_path.is_file():
            all_files.append(file_path)

    if not all_files:
        return {
            "total_files": 0,
            "total_size": 0,
            "avg_size": 0,
            "min_size": 0,
            "max_size": 0,
            "total_words": 0,
            "avg_words": 0,
            "file_types": {},
            "document_types": {},
            "completeness_types": {},
            "ai_classification_done": classify_with_ai,
            "analysis_timestamp": time.time(),
        }

    # Analyze files
    file_sizes = []
    word_counts = []
    file_types = {}
    document_types = {}
    completeness_types = {}

    # Collect documents for batch classification
    documents_to_classify = []
    file_info = []  # Store file info for later processing

    logger.info(f"Processing {len(all_files)} files...")

    # First pass: collect file info and extract text for classification
    for i, file_path in enumerate(all_files):
        if i % 100 == 0:
            logger.info(f"Processed {i}/{len(all_files)} files...")

        try:
            size = file_path.stat().st_size

            # Use file command to determine file type
            try:
                result = subprocess.run(
                    ["file", "-b", str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    file_output = result.stdout.strip().lower()
                    file_type = classify_file_type(file_output)
                else:
                    file_type = "unknown"
            except (
                subprocess.TimeoutExpired,
                subprocess.SubprocessError,
                FileNotFoundError,
            ):
                file_type = "unknown"

            # Count words in the file
            word_count = count_words_in_file(file_path, file_type)

            # Check for "openprescribing" in the full document text
            has_openprescribing = False
            if file_type in ["pdf", "html", "text", "docx", "office"]:
                text = extract_full_text(file_path, file_type)
                if text:
                    has_openprescribing = "openprescribing" in text.lower()

            # Store file info for later processing
            file_info.append(
                {
                    "path": file_path,
                    "size": size,
                    "file_type": file_type,
                    "word_count": word_count,
                    "has_openprescribing": has_openprescribing,
                }
            )

            # Collect documents for AI classification (only if successfully downloaded)
            if classify_with_ai and file_type in [
                "pdf",
                "html",
                "text",
                "docx",
                "office",
            ]:
                # Only classify if this file was successfully downloaded or if we don't have download status info
                if not successfully_downloaded_files or str(file_path) in successfully_downloaded_files:
                    text = extract_text_for_classification(file_path, file_type)
                    if text:
                        documents_to_classify.append((file_path, text))

        except (OSError, IOError):
            # Skip files we can't read
            continue

    # Batch AI classification
    ai_classifications = {}
    if classify_with_ai and documents_to_classify:
        # Process in batches of 50 (API limit consideration)
        batch_size = 50

        for i in range(0, len(documents_to_classify), batch_size):
            batch = documents_to_classify[i : i + batch_size]
            logger.info(
                f"Processing AI classification batch {i//batch_size + 1}/{(len(documents_to_classify)-1)//batch_size + 1}"
            )

            batch_results = classify_documents_batch(batch)
            ai_classifications.update(batch_results)
        
        # Update CSV with classification results
        if ai_classifications and csv_filename:
            update_csv_with_classifications(csv_filename, ai_classifications)

    # Collect OpenPrescribing detection data for CSV update
    openprescribing_data = {}
    for file_data in file_info:
        file_path = file_data["path"]
        has_openprescribing = file_data["has_openprescribing"]
        openprescribing_data[str(file_path)] = has_openprescribing

    # Update CSV with OpenPrescribing detection results
    if openprescribing_data and csv_filename:
        update_csv_with_openprescribing(csv_filename, openprescribing_data)

    # Second pass: aggregate statistics
    logger.info("Aggregating statistics...")

    openprescribing_count = 0

    for file_data in file_info:
        file_path = file_data["path"]
        size = file_data["size"]
        file_type = file_data["file_type"]
        word_count = file_data["word_count"]
        has_openprescribing = file_data["has_openprescribing"]

        file_sizes.append(size)
        word_counts.append(word_count)

        # Count OpenPrescribing mentions
        if has_openprescribing:
            openprescribing_count += 1

        # Get AI classification for this file
        ai_classification = ai_classifications.get(
            str(file_path),
            {"document_type": "unknown", "completeness": "unknown", "confidence": 0},
        )

        # Update file type statistics
        if file_type not in file_types:
            file_types[file_type] = {
                "count": 0,
                "total_size": 0,
                "total_words": 0,
                "openprescribing_count": 0,
                "sizes": [],
                "word_counts": [],
            }

        file_types[file_type]["count"] += 1
        file_types[file_type]["total_size"] += size
        file_types[file_type]["total_words"] += word_count
        if has_openprescribing:
            file_types[file_type]["openprescribing_count"] += 1
        file_types[file_type]["sizes"].append(size)
        file_types[file_type]["word_counts"].append(word_count)

        # Update AI classification statistics
        doc_type = ai_classification["document_type"]
        completeness = ai_classification["completeness"]

        if doc_type not in document_types:
            document_types[doc_type] = {
                "count": 0,
                "total_words": 0,
                "openprescribing_count": 0,
            }
        document_types[doc_type]["count"] += 1
        document_types[doc_type]["total_words"] += word_count
        if has_openprescribing:
            document_types[doc_type]["openprescribing_count"] += 1

        if completeness not in completeness_types:
            completeness_types[completeness] = {
                "count": 0,
                "total_words": 0,
                "openprescribing_count": 0,
            }
        completeness_types[completeness]["count"] += 1
        completeness_types[completeness]["total_words"] += word_count
        if has_openprescribing:
            completeness_types[completeness]["openprescribing_count"] += 1

    # Calculate averages for each file type
    for file_type in file_types:
        sizes = file_types[file_type]["sizes"]
        words = file_types[file_type]["word_counts"]

        if sizes:
            file_types[file_type]["avg_size"] = sum(sizes) / len(sizes)
        else:
            file_types[file_type]["avg_size"] = 0

        if words:
            file_types[file_type]["avg_words"] = sum(words) / len(words)
        else:
            file_types[file_type]["avg_words"] = 0

        # Remove the temporary lists to clean up the result
        del file_types[file_type]["sizes"]
        del file_types[file_type]["word_counts"]

    total_size = sum(file_sizes)
    total_words = sum(word_counts)
    avg_size = total_size / len(file_sizes) if file_sizes else 0
    avg_words = total_words / len(word_counts) if word_counts else 0

    result = {
        "total_files": len(file_sizes),
        "total_size": total_size,
        "avg_size": avg_size,
        "min_size": min(file_sizes) if file_sizes else 0,
        "max_size": max(file_sizes) if file_sizes else 0,
        "total_words": total_words,
        "avg_words": avg_words,
        "openprescribing_count": openprescribing_count,
        "file_types": file_types,
        "document_types": document_types,
        "completeness_types": completeness_types,
        "ai_classification_done": classify_with_ai,
        "analysis_timestamp": time.time(),
    }

    # Save report to disk if requested
    if report_file:
        logger.info(f"Saving analysis report to {report_file}")
        try:
            with open(report_file, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save report: {e}")

    return result


def classify_file_type(file_output):
    """Classify file type based on file command output."""
    file_output = file_output.lower()

    # PDF files
    if "pdf" in file_output:
        return "pdf"

    # HTML files
    elif "html" in file_output or "xml" in file_output:
        return "html"

    # Microsoft Office documents
    elif any(
        x in file_output for x in ["microsoft", "word", "excel", "powerpoint", "office"]
    ):
        if "word" in file_output:
            return "docx"
        elif "excel" in file_output:
            return "xlsx"
        elif "powerpoint" in file_output:
            return "pptx"
        else:
            return "office"

    # ZIP files (could be office docs)
    elif "zip" in file_output and "archive" in file_output:
        return "zip"

    # Text files
    elif any(x in file_output for x in ["text", "ascii"]):
        if "csv" in file_output:
            return "csv"
        elif "json" in file_output:
            return "json"
        else:
            return "text"

    # Image files
    elif any(x in file_output for x in ["image", "jpeg", "png", "gif", "bitmap"]):
        if "jpeg" in file_output or "jpg" in file_output:
            return "jpeg"
        elif "png" in file_output:
            return "png"
        elif "gif" in file_output:
            return "gif"
        else:
            return "image"

    # Binary/executable files
    elif any(x in file_output for x in ["executable", "binary", "data"]):
        return "binary"

    # Default
    else:
        return "unknown"


def fetch_papers(args):
    """Fetch papers subcommand implementation."""
    # Find papers in Google Scholar but not in PubMed
    gs_only_papers = find_gs_only_papers()

    if gs_only_papers.empty:
        logger.info("No papers found in Google Scholar but not in PubMed")
        return

    # Prepare CSV output
    csv_filename = args.output or "gs_only_downloads.csv"

    # Load existing downloads for idempotency
    existing_downloads = load_existing_downloads(csv_filename)
    write_csv_header(csv_filename)

    # Filter out papers already downloaded
    papers_to_download = gs_only_papers[
        ~gs_only_papers["ArticleURL"].isin(existing_downloads)
        & gs_only_papers["ArticleURL"].notna()
        & (gs_only_papers["ArticleURL"] != "")
    ]

    skipped_count = len(gs_only_papers) - len(papers_to_download)

    logger.info(f"Found {len(existing_downloads)} existing downloads")
    logger.info(f"Skipping {skipped_count} already processed papers")
    logger.info(
        f"Starting concurrent downloads for {len(papers_to_download)} papers..."
    )

    if papers_to_download.empty:
        logger.info("No new papers to download")
        return

    # Run async downloads
    try:
        downloaded_count, failed_count = asyncio.run(
            download_papers_concurrently(
                papers_to_download,
                csv_filename,
                data_dir=args.data_dir,
                max_concurrent=args.concurrent,
                timeout=args.timeout,
                batch_size=args.batch_size,
            )
        )

        logger.info(
            f"Download complete. {downloaded_count} successful, {failed_count} failed, {skipped_count} skipped."
        )
        logger.info(f"Results written to {csv_filename}")

    except KeyboardInterrupt:
        logger.info("Download interrupted by user. Progress saved to CSV.")
    except Exception as e:
        logger.error(f"Download failed with error: {e}")
        raise


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Download papers that are in Google Scholar but not in PubMed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s fetch                    # Download with default settings
  %(prog)s fetch -c 20             # Use 20 concurrent downloads
  %(prog)s fetch -o my_papers.csv  # Save results to custom file
  %(prog)s fetch --data-dir ./downloads  # Save files to custom directory
  
  %(prog)s stats                    # Show stats for default CSV
  %(prog)s stats my_papers.csv      # Show stats for custom CSV
  %(prog)s stats --recent           # Include recent download activity
  %(prog)s stats --sample 10        # Classify 10 random docs for verification
  %(prog)s stats --classify         # Include AI document classification
  %(prog)s stats --classify --regenerate  # Force AI classification of all documents
        """.strip(),
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Fetch subcommand
    fetch_parser = subparsers.add_parser(
        "fetch", help="Download papers from Google Scholar that are not in PubMed"
    )
    fetch_parser.add_argument(
        "-c",
        "--concurrent",
        type=int,
        default=10,
        help="Number of concurrent downloads (default: 10)",
    )
    fetch_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output CSV filename (default: gs_only_downloads.csv)",
    )
    fetch_parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to save downloaded files (default: data)",
    )
    fetch_parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)",
    )
    fetch_parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)",
    )

    # Stats subcommand
    stats_parser = subparsers.add_parser(
        "stats", help="Show summary statistics from download CSV file"
    )
    stats_parser.add_argument(
        "csv_file",
        type=str,
        nargs="?",
        default="gs_only_downloads.csv",
        help="CSV file to analyze (default: gs_only_downloads.csv)",
    )
    stats_parser.add_argument(
        "--recent", action="store_true", help="Show recent download activity"
    )
    stats_parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing downloaded files (default: data)",
    )
    stats_parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate analysis report (ignore cached data)",
    )
    stats_parser.add_argument(
        "--classify",
        action="store_true",
        help="Use Claude AI to classify document types (requires ANTHROPIC_API_KEY)",
    )
    stats_parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Classify only N random documents for verification (shows individual results with file links)",
    )
    stats_parser.add_argument(
        "--show-classifications",
        action="store_true",
        help="Show individual AI classification results from CSV",
    )

    args = parser.parse_args()

    if args.command == "fetch":
        fetch_papers(args)
    elif args.command == "stats":
        stats_papers(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
