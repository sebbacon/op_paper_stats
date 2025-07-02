#!/usr/bin/env python3
"""
Paper comparison and analysis tool for OpenPrescribing research.
Generates an interactive HTML report comparing PubMed and Google Scholar datasets.
"""

import pandas as pd
import numpy as np
import re
import json
import hashlib
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from jinja2 import Template
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperComparator:
    def __init__(
        self, papers_csv: str, op_papers_csv: str, cache_dir: str = "crossref_cache"
    ):
        self.papers_csv = papers_csv
        self.op_papers_csv = op_papers_csv
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # CrossRef API settings
        self.crossref_base_url = "https://api.crossref.org/works/"
        self.crossref_delay = 1.0  # Rate limiting

    def normalize_doi(self, doi: str) -> str:
        """Normalize DOI by removing prefixes and converting to lowercase."""
        if pd.isna(doi) or not doi:
            return ""

        doi = str(doi).strip().lower()

        # Remove common prefixes
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

    def normalize_title(self, title: str) -> str:
        """Normalize title for matching by handling punctuation and whitespace variations.
        Uses only the first 18 words in lowercase."""
        if pd.isna(title) or not title:
            return ""

        title = str(title).strip().lower()

        # Remove trailing periods and other end punctuation
        title = re.sub(r"[.!?]+$", "", title)

        # Replace common punctuation with spaces but preserve structure
        # Keep apostrophes in contractions, replace other punctuation with spaces
        title = re.sub(r"[^\w\s\']", " ", title)

        # Normalize multiple spaces to single spaces
        title = re.sub(r"\s+", " ", title)

        # Take only the first 18 words
        words = title.strip().split()
        title = " ".join(words[:18])

        return title

    def create_match_key(self, doi: str, title: str) -> str:
        """Create a matching key using DOI if available, otherwise normalized title."""
        doi_norm = self.normalize_doi(doi)
        if doi_norm:
            return f"doi:{doi_norm}"
        else:
            title_norm = self.normalize_title(title)
            return f"title:{title_norm}" if title_norm else ""

    def load_and_normalize_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both CSV files and normalize the data."""
        logger.info("Loading PubMed dataset...")
        papers_df = pd.read_csv(self.papers_csv)

        logger.info("Loading Google Scholar dataset...")
        op_papers_df = pd.read_csv(self.op_papers_csv)

        # Remove citations from the GA stuff -- they are usually cruft
        op_papers_df = op_papers_df[op_papers_df["Type"] != "CITATION"]

        # Fill missing values first
        papers_df = papers_df.fillna("")
        op_papers_df = op_papers_df.fillna("")

        # Normalize DOIs and titles
        papers_df["doi_normalized"] = papers_df["DOI"].apply(self.normalize_doi)
        op_papers_df["doi_normalized"] = op_papers_df["DOI"].apply(self.normalize_doi)

        # Create match keys (DOI preferred, title fallback)
        papers_df["match_key"] = papers_df.apply(
            lambda row: self.create_match_key(row["DOI"], row["Title"]), axis=1
        )
        op_papers_df["match_key"] = op_papers_df.apply(
            lambda row: self.create_match_key(row["DOI"], row["Title"]), axis=1
        )

        # Remove rows with empty match keys (no DOI and no title)
        papers_df = papers_df[papers_df["match_key"] != ""]
        op_papers_df = op_papers_df[op_papers_df["match_key"] != ""]

        # De-duplicate within each file on match_key
        logger.info("De-duplicating PubMed dataset...")
        papers_df = papers_df.drop_duplicates(subset=["match_key"], keep="first")

        logger.info("De-duplicating Google Scholar dataset...")
        op_papers_df = op_papers_df.drop_duplicates(subset=["match_key"], keep="first")

        # Ensure core_paper column exists in papers_df
        if "core_paper" not in papers_df.columns:
            papers_df["core_paper"] = 0

        logger.info(f"Loaded {len(papers_df)} papers from PubMed")
        logger.info(f"Loaded {len(op_papers_df)} papers from Google Scholar")

        return papers_df, op_papers_df

    def tag_provenance(
        self, papers_df: pd.DataFrame, op_papers_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Tag records with provenance and merge datasets."""
        # Add source tags
        papers_df["source"] = "pubmed"
        op_papers_df["source"] = "google_scholar"

        # Create additional title-based match keys for cross-matching
        papers_df["title_match_key"] = papers_df["Title"].apply(
            lambda x: f"title:{self.normalize_title(x)}" if x else ""
        )
        op_papers_df["title_match_key"] = op_papers_df["Title"].apply(
            lambda x: f"title:{self.normalize_title(x)}" if x else ""
        )

        # Standardize column names for merging
        papers_columns = {
            "Authors": "authors",
            "Title": "title",
            "Source": "journal",
            "Year": "year",
            "DOI": "doi_original",
            "Citations": "citations",
            "doi_normalized": "doi_normalized",
            "match_key": "match_key",
            "title_match_key": "title_match_key",
            "core_paper": "core_paper",
            "source": "source",
        }

        op_papers_columns = {
            "Authors": "authors",
            "Title": "title",
            "Source": "journal",
            "Year": "year",
            "DOI": "doi_original",
            "Cites": "citations",
            "doi_normalized": "doi_normalized",
            "match_key": "match_key",
            "title_match_key": "title_match_key",
            "source": "source",
        }

        # Rename and select relevant columns
        papers_clean = papers_df.rename(columns=papers_columns)[
            list(papers_columns.values())
        ]
        op_papers_clean = op_papers_df.rename(columns=op_papers_columns)[
            list(op_papers_columns.values())
        ]

        # Add missing columns
        if "core_paper" not in op_papers_clean.columns:
            op_papers_clean["core_paper"] = 0

        # Combine datasets
        combined_df = pd.concat([papers_clean, op_papers_clean], ignore_index=True)

        # Enhanced matching: check both primary match_key and title_match_key
        def determine_provenance_and_match_method(row):
            # Find all rows that match either by primary key or title key
            primary_matches = combined_df[combined_df["match_key"] == row["match_key"]]
            title_matches = combined_df[
                (combined_df["title_match_key"] == row["title_match_key"])
                & (combined_df["title_match_key"] != "")
            ]

            # Determine if this was matched by title only
            title_only_match = (
                len(title_matches) > 1
                and len(primary_matches) == 1
                and row["title_match_key"] != ""
            )

            # Combine both types of matches
            all_matches = pd.concat([primary_matches, title_matches]).drop_duplicates()
            sources = set(all_matches["source"])

            if len(sources) >= 2:
                provenance = "both"
            elif "pubmed" in sources:
                provenance = "pubmed"
            elif "google_scholar" in sources:
                provenance = "google_scholar"
            else:
                provenance = "unknown"

            # Determine match method
            if title_only_match:
                match_method = "Title"
            elif row["match_key"].startswith("doi:"):
                match_method = "DOI"
            elif row["match_key"].startswith("title:"):
                match_method = "Title"
            else:
                match_method = "Unknown"

            return provenance, match_method

        # Apply the function and split results
        results = combined_df.apply(
            determine_provenance_and_match_method, axis=1, result_type="expand"
        )
        combined_df["in_source"] = results[0]
        combined_df["original_match_method"] = results[1]

        # For deduplication, prioritize records that appear in both sources
        combined_df["priority"] = combined_df["in_source"].map(
            {"both": 1, "pubmed": 2, "google_scholar": 3, "unknown": 4}
        )

        # Deduplicate by match_key first (for DOI-based matches)
        combined_df = combined_df.sort_values(
            ["match_key", "priority", "core_paper"], ascending=[True, True, False]
        )
        combined_df = combined_df.drop_duplicates(subset=["match_key"], keep="first")

        # Then deduplicate by title_match_key (for title-based matches)
        # Only deduplicate records that don't already have 'both' provenance
        combined_df = combined_df.sort_values(
            ["title_match_key", "priority", "core_paper"], ascending=[True, True, False]
        )
        title_mask = (combined_df["title_match_key"] != "") & (
            combined_df["in_source"] != "both"
        )
        if title_mask.any():
            non_both_df = combined_df[title_mask].drop_duplicates(
                subset=["title_match_key"], keep="first"
            )
            both_df = combined_df[~title_mask]
            combined_df = pd.concat([both_df, non_both_df], ignore_index=True)

        combined_df = combined_df.drop(columns=["priority"])
        logger.info(f"Combined dataset has {len(combined_df)} unique papers")

        return combined_df

    def identify_comparison_sets(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Identify intersection and gap sets."""
        sets = {
            "intersection": df[df["in_source"] == "both"].copy(),
            "pubmed_only": df[df["in_source"] == "pubmed"].copy(),
            "google_scholar_only": df[df["in_source"] == "google_scholar"].copy(),
        }

        logger.info(f"Intersection: {len(sets['intersection'])} papers")
        logger.info(f"PubMed-only gap: {len(sets['pubmed_only'])} papers")
        logger.info(
            f"Google Scholar-only gap: {len(sets['google_scholar_only'])} papers"
        )

        return sets

    def get_crossref_metadata(self, doi: str) -> Optional[Dict]:
        """Fetch metadata from CrossRef API with caching."""
        if not doi or pd.isna(doi):
            return None

        doi = str(doi)  # Ensure it's a string
        cache_file = self.cache_dir / f"{hashlib.md5(doi.encode()).hexdigest()}.json"

        # Check cache first
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except:
                pass

        # Fetch from API
        try:
            time.sleep(self.crossref_delay)  # Rate limiting
            response = requests.get(
                f"{self.crossref_base_url}{doi}",
                headers={"User-Agent": "PaperComparator/1.0"},
            )

            if response.status_code == 200:
                data = response.json()
                # Cache the result
                with open(cache_file, "w") as f:
                    json.dump(data, f)
                return data
            else:
                logger.warning(
                    f"CrossRef API returned {response.status_code} for DOI: {doi}"
                )
                return None

        except Exception as e:
            logger.warning(f"Error fetching CrossRef data for {doi}: {e}")
            return None

    def enrich_with_crossref(
        self, df: pd.DataFrame, max_requests: int = 50
    ) -> pd.DataFrame:
        """Enrich dataset with CrossRef metadata."""
        logger.info(
            f"Enriching with CrossRef metadata (max {max_requests} requests)..."
        )

        # Add empty columns for CrossRef data
        df["crossref_journal"] = ""
        df["crossref_abstract"] = ""
        df["crossref_subjects"] = ""
        df["crossref_author_count"] = 0
        df["crossref_issn"] = ""
        df["crossref_pub_date"] = ""

        # Process a subset of DOIs to avoid hitting API limits
        dois_to_process = df[df["doi_normalized"] != ""]["doi_normalized"].unique()[
            :max_requests
        ]

        for i, doi in enumerate(dois_to_process):
            if i % 10 == 0:
                logger.info(f"Processing DOI {i+1}/{len(dois_to_process)}")

            metadata = self.get_crossref_metadata(doi)
            if metadata and "message" in metadata:
                work = metadata["message"]

                # Extract relevant fields
                journal = ""
                if "container-title" in work and work["container-title"]:
                    journal = work["container-title"][0]

                abstract = work.get("abstract", "")
                if abstract:
                    abstract = (
                        abstract[:500] + "..." if len(abstract) > 500 else abstract
                    )

                subjects = []
                if "subject" in work:
                    subjects = work["subject"][:3]  # Limit to first 3 subjects

                author_count = len(work.get("author", []))

                issn = ""
                if "ISSN" in work and work["ISSN"]:
                    issn = work["ISSN"][0]

                pub_date = ""
                if "published-print" in work:
                    date_parts = work["published-print"].get("date-parts", [[]])
                    if date_parts and date_parts[0]:
                        pub_date = "-".join(map(str, date_parts[0]))

                # Update dataframe
                mask = df["doi_normalized"] == doi
                df.loc[mask, "crossref_journal"] = journal
                df.loc[mask, "crossref_abstract"] = abstract
                df.loc[mask, "crossref_subjects"] = ", ".join(subjects)
                df.loc[mask, "crossref_author_count"] = author_count
                df.loc[mask, "crossref_issn"] = issn
                df.loc[mask, "crossref_pub_date"] = pub_date

        return df

    def compute_analytical_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute additional analytical fields."""
        # Convert year to numeric
        df["year_numeric"] = pd.to_numeric(df["year"], errors="coerce")

        # Create year buckets
        def year_bucket(year):
            if pd.isna(year):
                return "Unknown"
            year = int(year)
            if year <= 2010:
                return "≤2010"
            elif year <= 2015:
                return "2011-2015"
            elif year <= 2020:
                return "2016-2020"
            else:
                return "≥2021"

        df["year_bucket"] = df["year_numeric"].apply(year_bucket)

        # Unified citation count
        df["citations_numeric"] = pd.to_numeric(
            df["citations"], errors="coerce"
        ).fillna(0)

        # Use the original match method if available, otherwise derive from match_key
        if "original_match_method" in df.columns:
            df["match_type"] = df["original_match_method"]
        else:

            def get_match_type(x):
                if pd.isna(x) or not x:
                    return "Unknown"
                x = str(x)
                if x.startswith("doi:"):
                    return "DOI"
                elif x.startswith("title:"):
                    return "Title"
                else:
                    return "Unknown"

            df["match_type"] = df["match_key"].apply(get_match_type)

        # Boolean flags
        df["is_core_paper"] = df["core_paper"] == 1
        df["has_abstract"] = df["crossref_abstract"] != ""
        df["is_likely_false_positive"] = None  # To be filled by curator

        return df

    def generate_html_report(
        self, df: pd.DataFrame, comparison_sets: Dict[str, pd.DataFrame]
    ) -> str:
        """Generate the interactive HTML report."""

        # Prepare data for JSON embedding
        report_data = {
            "all_papers": df.to_dict("records"),
            "comparison_sets": {
                k: v.to_dict("records") for k, v in comparison_sets.items()
            },
            "summary_stats": {
                "total_papers": len(df),
                "intersection_count": len(comparison_sets["intersection"]),
                "pubmed_only_count": len(comparison_sets["pubmed_only"]),
                "google_scholar_only_count": len(
                    comparison_sets["google_scholar_only"]
                ),
                "core_papers_count": len(df[df["is_core_paper"]]),
                "year_distribution": df["year_bucket"].value_counts().to_dict(),
                "source_distribution": df["in_source"].value_counts().to_dict(),
                "match_type_distribution": df["match_type"].value_counts().to_dict(),
            },
            "data_hash": hashlib.sha256(df.to_csv().encode()).hexdigest()[:16],
        }

        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paper Comparison Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .dashboard { display: flex; gap: 20px; margin-bottom: 30px; padding: 20px; background: #f8f9fa; border-radius: 6px; }
        .stat-card { flex: 1; text-align: center; padding: 15px; background: white; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .stat-number { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .stat-label { color: #666; font-size: 0.9em; }
        .controls { margin-bottom: 20px; display: flex; gap: 15px; align-items: center; flex-wrap: wrap; }
        .control-group { display: flex; align-items: center; gap: 5px; }
        .btn { padding: 8px 16px; border: 1px solid #ddd; background: white; border-radius: 4px; cursor: pointer; }
        .btn.active { background: #007bff; color: white; border-color: #007bff; }
        .search-box { padding: 8px; border: 1px solid #ddd; border-radius: 4px; width: 200px; }
        .table-container { overflow-x: auto; border: 1px solid #ddd; border-radius: 6px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; position: sticky; top: 0; }
        .badge { padding: 3px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 500; }
        .badge-both { background: #d4edda; color: #155724; }
        .badge-pubmed { background: #cce5ff; color: #004085; }
        .badge-google-scholar { background: #fff3cd; color: #856404; }
        .badge-core { background: #f8d7da; color: #721c24; }
        .doi-link { color: #007bff; text-decoration: none; }
        .doi-link:hover { text-decoration: underline; }
        .row-expandable { cursor: pointer; }
        .row-expandable:hover { background: #f8f9fa; }
        .details-row { background: #f8f9fa; }
        .details-content { padding: 20px; }
        .hidden { display: none; }
        .year-chart, .journal-chart { margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 6px; }
        .chart-title { font-weight: 600; margin-bottom: 10px; }
        .export-buttons { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Paper Comparison Report</h1>
        <p>Comparing PubMed and Google Scholar datasets</p>
        
        <div class="dashboard" id="dashboard">
            <div class="stat-card">
                <div class="stat-number" id="total-count">{{ data.summary_stats.total_papers }}</div>
                <div class="stat-label">Total Papers</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="intersection-count">{{ data.summary_stats.intersection_count }}</div>
                <div class="stat-label">In Both Sets</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="pubmed-only-count">{{ data.summary_stats.pubmed_only_count }}</div>
                <div class="stat-label">PubMed Only</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="google-scholar-only-count">{{ data.summary_stats.google_scholar_only_count }}</div>
                <div class="stat-label">Google Scholar Only</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="core-count">{{ data.summary_stats.core_papers_count }}</div>
                <div class="stat-label">Core Papers</div>
            </div>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label>Source Filter:</label>
                <button class="btn active" data-filter="all">All</button>
                <button class="btn" data-filter="both">Both</button>
                <button class="btn" data-filter="pubmed">PubMed Only</button>
                <button class="btn" data-filter="google_scholar">Google Scholar Only</button>
            </div>
            <div class="control-group">
                <label>Show Core Papers Only:</label>
                <input type="checkbox" id="core-only-toggle">
            </div>
            <div class="control-group">
                <input type="text" class="search-box" id="search-box" placeholder="Search titles, authors, DOIs...">
            </div>
        </div>
        
        <div class="table-container">
            <table id="papers-table">
                <thead>
                    <tr>
                        <th>DOI</th>
                        <th>Title</th>
                        <th>Authors</th>
                        <th>Year</th>
                        <th>Journal</th>
                        <th>Source</th>
                        <th>Match</th>
                        <th>Core</th>
                        <th>Citations</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="papers-tbody">
                </tbody>
            </table>
        </div>
        
        <div class="export-buttons">
            <button class="btn" onclick="exportDOIs()">Export DOI List</button>
            <button class="btn" onclick="exportCSV()">Export Current View as CSV</button>
        </div>
    </div>

    <script>
        // Embed data
        const reportData = {{ data_json | safe }};
        let currentFilter = 'all';
        let coreOnlyMode = false;
        let searchTerm = '';
        let filteredData = [...reportData.all_papers];

        function renderTable() {
            const tbody = document.getElementById('papers-tbody');
            tbody.innerHTML = '';
            
            filteredData.forEach((paper, index) => {
                const row = document.createElement('tr');
                row.className = 'row-expandable';
                row.onclick = () => toggleDetails(index);
                
                const doiLink = paper.doi_normalized ? 
                    `<a href="https://doi.org/${paper.doi_normalized}" target="_blank" class="doi-link">${paper.doi_normalized}</a>` : 
                    'N/A';
                
                const sourceBadge = `<span class="badge badge-${paper.in_source === 'both' ? 'both' : paper.in_source === 'pubmed' ? 'pubmed' : 'google-scholar'}">${paper.in_source === 'google_scholar' ? 'Google Scholar' : paper.in_source}</span>`;
                const matchBadge = `<span class="badge ${paper.match_type === 'DOI' ? 'badge-both' : 'badge-op'}">${paper.match_type}</span>`;
                const coreBadge = paper.is_core_paper ? '<span class="badge badge-core">Core</span>' : '';
                
                row.innerHTML = `
                    <td>${doiLink}</td>
                    <td>${paper.title || 'N/A'}</td>
                    <td>${paper.authors || 'N/A'}</td>
                    <td>${paper.year || 'N/A'}</td>
                    <td>${paper.journal || paper.crossref_journal || 'N/A'}</td>
                    <td>${sourceBadge}</td>
                    <td>${matchBadge}</td>
                    <td>${coreBadge}</td>
                    <td>${paper.citations_numeric || 0}</td>
                    <td><button class="btn" onclick="event.stopPropagation(); toggleDetails(${index})">Details</button></td>
                `;
                
                tbody.appendChild(row);
                
                // Add details row (initially hidden)
                const detailsRow = document.createElement('tr');
                detailsRow.className = 'details-row hidden';
                detailsRow.id = `details-${index}`;
                detailsRow.innerHTML = `
                    <td colspan="10">
                        <div class="details-content">
                            <h4>Paper Details</h4>
                            <p><strong>Match Type:</strong> ${paper.match_type} (${paper.match_key.substring(0, 50)}...)</p>
                            <p><strong>Abstract:</strong> ${paper.crossref_abstract || 'Not available'}</p>
                            <p><strong>Subjects:</strong> ${paper.crossref_subjects || 'Not available'}</p>
                            <p><strong>Author Count:</strong> ${paper.crossref_author_count || 'Unknown'}</p>
                            <p><strong>ISSN:</strong> ${paper.crossref_issn || 'Not available'}</p>
                            <p><strong>Publication Date:</strong> ${paper.crossref_pub_date || paper.year || 'Unknown'}</p>
                            <p><strong>Year Bucket:</strong> ${paper.year_bucket}</p>
                            <div>
                                <strong>External Links:</strong>
                                <a href="https://scholar.google.com/scholar?q=${encodeURIComponent(paper.title)}" target="_blank">Google Scholar</a> |
                                <a href="https://pubmed.ncbi.nlm.nih.gov/?term=${encodeURIComponent(paper.title)}" target="_blank">PubMed</a>
                                ${paper.doi_normalized ? `| <a href="https://unpaywall.org/${paper.doi_normalized}" target="_blank">Unpaywall</a>` : ''}
                            </div>
                        </div>
                    </td>
                `;
                tbody.appendChild(detailsRow);
            });
        }
        
        function toggleDetails(index) {
            const detailsRow = document.getElementById(`details-${index}`);
            detailsRow.classList.toggle('hidden');
        }
        
        function filterData() {
            filteredData = reportData.all_papers.filter(paper => {
                // Source filter
                if (currentFilter !== 'all' && paper.in_source !== currentFilter) {
                    return false;
                }
                
                // Core papers filter
                if (coreOnlyMode && !paper.is_core_paper) {
                    return false;
                }
                
                // Search filter
                if (searchTerm) {
                    const searchFields = [
                        paper.title,
                        paper.authors,
                        paper.doi_normalized,
                        paper.journal,
                        paper.crossref_journal
                    ].join(' ').toLowerCase();
                    
                    if (!searchFields.includes(searchTerm.toLowerCase())) {
                        return false;
                    }
                }
                
                return true;
            });
            
            renderTable();
            updateCounts();
        }
        
        function updateCounts() {
            document.getElementById('total-count').textContent = filteredData.length;
        }
        
        function exportDOIs() {
            const dois = filteredData
                .filter(paper => paper.doi_normalized)
                .map(paper => paper.doi_normalized)
                .join('\\n');
            
            navigator.clipboard.writeText(dois).then(() => {
                alert('DOI list copied to clipboard!');
            });
        }
        
        function exportCSV() {
            const headers = ['doi_normalized', 'title', 'authors', 'year', 'journal', 'in_source', 'is_core_paper', 'citations_numeric'];
            const csv = [headers.join(',')];
            
            filteredData.forEach(paper => {
                const row = headers.map(header => {
                    const value = paper[header] || '';
                    return `"${String(value).replace(/"/g, '""')}"`;
                });
                csv.push(row.join(','));
            });
            
            const blob = new Blob([csv.join('\\n')], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'filtered_papers.csv';
            a.click();
        }
        
        // Event listeners
        document.querySelectorAll('[data-filter]').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('[data-filter]').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentFilter = btn.dataset.filter;
                filterData();
            });
        });
        
        document.getElementById('core-only-toggle').addEventListener('change', (e) => {
            coreOnlyMode = e.target.checked;
            filterData();
        });
        
        document.getElementById('search-box').addEventListener('input', (e) => {
            searchTerm = e.target.value;
            filterData();
        });
        
        // Initial render
        renderTable();
    </script>
</body>
</html>
        """

        template = Template(html_template)
        return template.render(
            data=report_data, data_json=json.dumps(report_data, default=str)
        )

    def run_analysis(self, output_file: str = "paper_comparison_report.html"):
        """Run the complete analysis pipeline."""
        logger.info("Starting paper comparison analysis...")

        # Step 1: Load and normalize data
        papers_df, op_papers_df = self.load_and_normalize_data()

        # Step 2: Tag provenance and merge
        combined_df = self.tag_provenance(papers_df, op_papers_df)

        # Step 3: Identify comparison sets
        comparison_sets = self.identify_comparison_sets(combined_df)

        # Step 4: Enrich with CrossRef metadata (limited requests)
        combined_df = self.enrich_with_crossref(combined_df, max_requests=50)

        # Step 5: Compute analytical fields
        combined_df = self.compute_analytical_fields(combined_df)

        # Step 6: Generate HTML report
        html_report = self.generate_html_report(combined_df, comparison_sets)

        # Write report to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_report)

        logger.info(f"Report generated: {output_file}")
        return combined_df, comparison_sets


if __name__ == "__main__":
    comparator = PaperComparator("papers.csv", "op_papers.csv")
    df, sets = comparator.run_analysis()
