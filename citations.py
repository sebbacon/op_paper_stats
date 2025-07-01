import argparse
import requests
import csv
import time
from tqdm import tqdm

query = "openprescribing"
retmax = 500
email = "seb.bacon@gmail.com"

API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def search_ids(db, term):
    resp = requests.get(
        f"{API}esearch.fcgi",
        params={
            "db": db,
            "term": term,
            "retmax": retmax,
            "retmode": "json",
            "email": email,
        },
    )
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]


def get_summary(db, ids):
    if not ids:
        return {}
    resp = requests.get(
        f"{API}esummary.fcgi",
        params={
            "db": db,
            "id": ",".join(ids),
            "retmode": "json",
            "email": email,
        },
    )
    resp.raise_for_status()
    return resp.json()["result"]


def convert_pmcid_to_pmid(pmcids):
    if not pmcids:
        return {}
    resp = requests.get(
        f"{API}elink.fcgi",
        params={
            "dbfrom": "pmc",
            "db": "pubmed",
            "id": ",".join(pmcids),
            "retmode": "json",
            "email": email,
            "linkname": "pmc_pubmed",  # or pmc_refs_pubmed -- things that _cite_ it
        },
    )
    resp.raise_for_status()
    links = resp.json()["linksets"][0]["linksetdbs"][0]["links"]
    return links


def get_citation_counts(pmids):
    counts = {}
    for pmid in tqdm(pmids, desc="Fetching citations"):
        resp = requests.get(
            f"{API}elink.fcgi",
            params={
                "dbfrom": "pubmed",
                "linkname": "pubmed_pubmed_citedin",
                "id": pmid,
                "retmode": "json",
                "email": email,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        linksets = data.get("linksets", [])
        if linksets and "linksetdbs" in linksets[0]:
            count = len(linksets[0]["linksetdbs"][0].get("links", []))
            counts[pmid] = count
        else:
            counts[pmid] = 0
        time.sleep(0.34)  # NCBI rate limit ~3 requests/sec
    return counts


def make_csv(output_file="papers.csv"):
    # Step 1: PMC full-text search
    pmc_ids = search_ids("pmc", f"{query} [Body - All Words]")
    pmc_pmids = convert_pmcid_to_pmid(pmc_ids)

    # Step 2: PubMed metadata-only search (excluding PMC)
    pubmed_ids = search_ids("pubmed", f"{query} NOT pubmed pmc[sb]")

    # Step 3: Combine and deduplicate
    all_pmids = list(set(pmc_pmids + pubmed_ids))

    # Step 4: Fetch summaries
    results = get_summary("pubmed", all_pmids)

    # Step 5: get citation counts

    citation_counts = get_citation_counts(all_pmids)

    # Step 5: Write CSV
    with open("papers.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Authors",
                "Title",
                "Source",
                "Year",
                "Volume",
                "Issue",
                "Pages",
                "DOI",
                "Citations",
                "core_paper",
            ]
        )
        for pmid in all_pmids:
            item = results.get(pmid)
            if not item or item.get("uid") == "0":
                continue
            title = item.get("title", "")
            authors = "; ".join([a["name"] for a in item.get("authors", [])])
            journal = item.get("source", "")
            pubdate = item.get("data", "")
            year = pubdate[:4] if pubdate else ""
            volume = item.get("volume", "")
            issue = item.get("issue", "")
            pages = item.get("pages", "")
            doi = item.get("elocationid", "")
            cites = citation_counts.get(pmid, None)
            writer.writerow(
                [authors, title, journal, year, volume, issue, pages, doi, cites, ""]
            )


def format_jama_citation(row):
    """Formats a citation row into JAMA-style markdown"""
    authors = "; ".join([a.strip() for a in row["Authors"].split(";")])
    title = row["Title"].rstrip(".")
    journal = f"*{row['Source']}*" if row["Source"] else ""
    year = row["Year"]
    volume = row["Volume"]
    issue = f"({row['Issue']})" if row["Issue"] else ""
    pages = f":{row['Pages']}" if row["Pages"] else ""
    doi = f"https://doi.org/{row['DOI']}" if row["DOI"] else ""

    return (
        (f"- {authors}. {title}. {journal} {year};{volume}{issue}{pages}. {doi}")
        .replace("  ", " ")
        .replace(" .", ".")
        .strip()
    )


def format_metrics_as_md_table(metrics):
    """Formats metrics dictionary as markdown table"""
    metrics_display = [
        ("Papers", "Papers"),
        ("Citations", "Total Citations"),
        ("Years", "Time Span (Years)"),
        ("Cites per Year", "Citations/Year"),
        ("Cites per Paper", "Citations/Paper"),
        ("h-index", "h-index"),
        ("g-index", "g-index"),
        ("hl-index", "hl-index"),
        ("Most Cited Paper", "Most Cited Paper"),
        ("Median Citations", "Median Citations"),
    ]

    headers = ["Metric", "Core Papers", "Citing Papers", "All Papers"]
    rows = []

    for metric_key, display_name in metrics_display:
        core_val = metrics["core_papers"].get(metric_key, "")
        citing_val = metrics["citing_papers"].get(metric_key, "")
        all_val = metrics["all_papers"].get(metric_key, "")

        rows.append(
            [
                display_name,
                _format_value(core_val),
                _format_value(citing_val),
                _format_value(all_val),
            ]
        )

    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        table += "| " + " | ".join(row) + " |\n"

    return table


def _format_value(value):
    """Formats numeric values for table display"""
    if isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def compute_metrics_from_csv(filepath):
    import sys

    # Read CSV and check for core_paper column
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "core_paper" not in reader.fieldnames:
            sys.exit(
                "Error: CSV missing 'core_paper' column. Please add column marking core papers (True/False)."
            )

        rows = list(reader)
        # Validate core_paper values
        required_values = {"true", "1", "yes", "false", "0", "no"}
        bad_rows = [
            (i + 1, row["core_paper"])
            for i, row in enumerate(rows)
            if str(row["core_paper"]).strip().lower() not in required_values
        ]

        if bad_rows:
            err_msg = "Invalid core_paper values found. Please fix these rows:\n"
            err_msg += "\n".join(
                f"Row {row[0]}: '{row[1]}' (valid values: True/False/Yes/No/1/0)"
                for row in bad_rows
            )
            sys.exit(err_msg)

    def _compute_metrics(subset_rows):
        citations = []
        years = []
        coauthor_counts = []

        for row in subset_rows:
            try:
                cites = int(row["Citations"]) if row["Citations"] else 0
                citations.append(cites)

                year = int(row["Year"]) if row["Year"] else None
                if year:
                    years.append(year)

                authors = row["Authors"].split(";")
                coauthor_counts.append(len(authors))
            except Exception as e:
                print(f"Skipping row due to error: {e}")
                continue

        total_papers = len(citations)
        total_citations = sum(citations)
        span_years = max(years) - min(years) + 1 if years else 0
        cites_per_year = total_citations / span_years if span_years else 0
        cites_per_paper = total_citations / total_papers if total_papers else 0

        sorted_citations = sorted(citations, reverse=True)

        # h-index
        h_index = sum(c >= i + 1 for i, c in enumerate(sorted_citations))

        # g-index
        cumulative = 0
        g_index = 0
        for i, c in enumerate(sorted_citations):
            cumulative += c
            if cumulative >= (i + 1) ** 2:
                g_index = i + 1
            else:
                break

        # hl-index: weighted h-index for co-authorship
        hl_numerator = 0
        hl_denominator = 0
        for c, a in zip(citations, coauthor_counts):
            if a > 0:
                hl_numerator += c / a
                hl_denominator += 1
        hl_index = hl_numerator / hl_denominator if hl_denominator else 0

        return {
            "Papers": total_papers,
            "Citations": total_citations,
            "Years": span_years,
            "Cites per Year": round(cites_per_year, 2),
            "Cites per Paper": round(cites_per_paper, 2),
            "h-index": h_index,
            "g-index": g_index,
            "hl-index": round(hl_index, 2),
            "Most Cited Paper": max(citations) if citations else 0,
            "Median Citations": (
                sorted(citations)[len(citations) // 2] if citations else 0
            ),
        }

    # Split into core papers and citing papers
    core_papers = [
        r for r in rows if r["core_paper"].strip().lower() in ["true", "1", "yes"]
    ]
    citing_papers = [
        r for r in rows if r["core_paper"].strip().lower() in ["false", "0", "no", ""]
    ]

    return {
        "core_papers": _compute_metrics(core_papers),
        "citing_papers": _compute_metrics(citing_papers),
        "all_papers": _compute_metrics(rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Manage citation data and metrics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Generate CSV from PubMed data")

    # Metrics command
    metrics_parser = subparsers.add_parser(
        "metrics", help="Compute metrics from existing CSV"
    )
    metrics_parser.add_argument("--input", required=True, help="Input CSV file path")

    # List core papers command
    list_core_parser = subparsers.add_parser(
        "list-core", help="Generate markdown list of core papers in JAMA format"
    )
    list_core_parser.add_argument("--input", required=True, help="Input CSV file path")
    list_core_parser.add_argument(
        "--output", help="Output markdown file path (prints to console if omitted)"
    )

    args = parser.parse_args()

    if args.command == "fetch":
        make_csv("papers.csv")
    elif args.command == "metrics":
        metrics = compute_metrics_from_csv(args.input)
        print("# Citations metrics from Pubmed")
        print(format_metrics_as_md_table(metrics))
    elif args.command == "list-core":
        metrics = compute_metrics_from_csv(args.input)
        breakpoint()
        # Sort papers by year descending (newest first), then title ascending
        sorted_rows = sorted(
            metrics["core_papers"],
            key=lambda row: (
                -int(row["Year"]) if row["Year"].strip().isdigit() else 0,
                row["Title"].lower(),
            ),
        )

        citations = [format_jama_citation(row) for row in sorted_rows]
        output = "\n".join(citations)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
        else:
            print(output)


if __name__ == "__main__":
    main()
