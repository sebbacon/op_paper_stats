import argparse
import os
import re
import subprocess
import json
from dotenv import load_dotenv
import requests
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import statistics
from collections import defaultdict
from google.cloud import bigquery
from google.oauth2 import service_account


def print_summary_statistics(credentials_path: str, year: int, month: int) -> None:
    """Execute summary statistics query and print results as markdown table"""
    if not os.path.exists(credentials_path):
        raise RuntimeError(f"Credentials file {credentials_path} not found")

    credentials = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=["https://www.googleapis.com/auth/bigquery"]
    )
    client = bigquery.Client(credentials=credentials)

    sql = f"""
    WITH base AS (
      SELECT
        month,
        EXTRACT(YEAR FROM month) AS year,
        FORMAT_DATE('%Y-%m', DATE(month)) AS year_month,
        items,
        net_cost,
        actual_cost,
        quantity,
        sha,
        practice,
        bnf_code
      FROM ebmdatalab.hscic.normalised_prescribing
    ),
    latest_month AS (
      SELECT DATE({year}, {month}, 1) AS analysis_month
    ),
    labelled_data AS (
      SELECT
        CAST(year AS STRING) AS label,
        items,
        net_cost,
        actual_cost,
        quantity,
        sha,
        practice,
        bnf_code
      FROM base

      UNION ALL

      SELECT
        'All Time',
        items,
        net_cost,
        actual_cost,
        quantity,
        sha,
        practice,
        bnf_code
      FROM base

      UNION ALL

      SELECT
        CONCAT('Month: ', '{year}-{month:02}'),
        b.items,
        b.net_cost,
        b.actual_cost,
        b.quantity,
        b.sha,
        b.practice,
        b.bnf_code
      FROM base b
      JOIN latest_month lm ON CAST(b.month AS DATE) = lm.analysis_month
      WHERE EXTRACT(YEAR FROM b.month) = {year}
        AND EXTRACT(MONTH FROM b.month) = {month}
    )

    SELECT
      label,
      COUNT(*) AS total_rows,
      SUM(items) AS total_items,
      SUM(net_cost) AS total_net_cost,
      SUM(actual_cost) AS total_actual_cost,
      SUM(quantity) AS total_quantity,
      COUNT(DISTINCT sha) AS distinct_shas,
      COUNT(DISTINCT practice) AS distinct_practices,
      COUNT(DISTINCT bnf_code) AS distinct_bnf_codes
    FROM labelled_data
    GROUP BY label
    ORDER BY label
    """

    try:
        query_job = client.query(sql)
        results = query_job.result()

        print(f"\n## Prescribing Data Summary ({year}-{month:02})\n")
        columns = [field.name for field in results.schema]
        print("| " + " | ".join(columns) + " |")
        print("| " + " | ".join(["---"] * len(columns)) + " |")

        for row in results:
            values = []
            for col in columns:
                value = getattr(row, col)
                if col == "label":
                    formatted = str(value)
                elif col in ["total_net_cost", "total_actual_cost"]:
                    formatted = f"£{value:,.2f}"
                elif col == "total_quantity":
                    formatted = f"{value:,.0f}"
                else:  # Handle counts
                    formatted = f"{value:,}"
                values.append(formatted)
            print("| " + " | ".join(values) + " |")

    except Exception as e:
        raise RuntimeError(f"Query failed: {str(e)}")


def fetch_characteristics(
    credentials_path,
    year: int,  # Add year parameter
    month: int,  # Add month parameter
    monthly_table=None,  # Remove default value
):
    """Fetch table metadata from BigQuery using service account credentials"""
    if monthly_table is None:
        monthly_table = f"ebmdatalab.hscic.prescribing_{year}_{month:02d}"
    if not os.path.exists(credentials_path):
        raise RuntimeError(
            f"Credentials file {credentials_path} not found\n"
            "1. Go to GCP Console > IAM & Admin > Service Accounts\n"
            "2. Create service account with 'BigQuery Metadata Viewer' role\n"
            "3. Generate JSON key and save it as gcp-credentials.json"
        )

    try:
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=["https://www.googleapis.com/auth/bigquery"]
        )
        client = bigquery.Client(credentials=credentials)

        # Get main table reference
        main_table = client.get_table("ebmdatalab.hscic.prescribing_v1")
        print(f"All raw rows ever recorded: {main_table.num_rows:,}")
        print(f"Physical storage footprint: {main_table.num_bytes / 1e9:.2f} GB\n")

        # Get monthly table reference
        monthly_table_ref = client.get_table(monthly_table)
        print(f"Most recent monthly table ({monthly_table}):")
        print(f"  Rows this month: {monthly_table_ref.num_rows:,}")
        print(f"  Monthly storage: {monthly_table_ref.num_bytes / 1e9:.2f} GB")

        # Get physical download size
        print("\nRaw CSV download size:")
        try:
            cmd = [
                "curl",
                "-sL",
                "--range",
                "0-0",
                "-D",
                "-",
                f"https://opendata.nhsbsa.net/dataset/65050ec0-5abd-48ce-989d-defc08ed837e/resource/7e7196a0-8fc8-4539-8322-ebd0d6554463/download/epd_{year}{month:02d}.csv",
                "|",
                "awk",
                '\'/Content-Range/ { split($3, a, "/"); printf "%.2f GB\\n", a[2] / (1024^3) }\'',
            ]
            result = subprocess.run(
                " ".join(cmd), shell=True, check=True, capture_output=True, text=True
            )
            size = result.stdout.strip()
            print(f"  {size} (via HTTP HEAD)")
        except subprocess.CalledProcessError as e:
            print(f"  Error checking download size: {str(e)}")
            if e.stderr:
                print(f"  STDERR: {e.stderr.strip()}")

    except Exception as e:
        raise RuntimeError(f"Failed to access BigQuery: {str(e)}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Manage paper characteristics data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add new subcommands
    measures_parser = subparsers.add_parser(
        "measures", help="Analyze measure definitions from OpenPrescribing repo"
    )

    bigquery_parser = subparsers.add_parser(
        "bigquery", help="Fetch BigQuery table characteristics"
    )
    summary_parser = subparsers.add_parser(
        "summary", help="Print prescribing summary statistics"
    )
    webstats_parser = subparsers.add_parser(
        "webstats", help="Show website traffic statistics"
    )
    webstats_parser.add_argument(
        "--app-name",
        nargs="*",
        default=[
            "measures",
            "price_per_unit",
            "ghost_generics",
            "concessions",
            "analyse",
        ],
        help="Filter stats to URLs containing these app names (space-separated)",
    )
    all_parser = subparsers.add_parser(
        "all", help="Run all reports: bigquery, summary, webstats"
    )

    # Add common arguments to relevant subcommands
    for p in [bigquery_parser, summary_parser, all_parser]:
        p.add_argument(
            "--credentials",
            default="gcp-credentials.json",
            help="Path to GCP service account JSON credentials",
        )
        p.add_argument(
            "--year",
            type=int,
            help="Year to analyze (default: 3 months ago)",
        )
        p.add_argument(
            "--month",
            type=int,
            help="Month to analyze (1-12, default: 3 months ago)",
        )

    args = parser.parse_args()

    # Common year/month calculation logic
    def get_year_month():
        now = datetime.now()
        if (
            not hasattr(args, "year")
            or not hasattr(args, "month")
            or args.year is None
            or args.month is None
        ):
            three_months_ago = now - relativedelta(months=3)
            year = args.year if args.year is not None else three_months_ago.year
            month = args.month if args.month is not None else three_months_ago.month
            return year, month
        return args.year, args.month

    # Command handling
    if args.command == "measures":
        repo_path = Path.cwd() / "openprescribing-repo"

        # Clone or update existing repo
        if repo_path.exists():
            print(f"Updating existing repository at {repo_path}")
            try:
                subprocess.run(
                    ["git", "-C", str(repo_path), "pull", "--ff-only"],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to update repository:\n" f"{e.stderr.decode().strip()}"
                )
        else:
            print(f"Cloning repository to {repo_path}")
            try:
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth=1",
                        "https://github.com/bennettoxford/openprescribing.git",
                        str(repo_path),
                    ],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to clone repository:\n" f"{e.stderr.decode().strip()}"
                )

        tags_def = fetch_measure_tags(repo_path)
        analysis = process_measures(repo_path, tags_def)
        print_measures_report(analysis)
    elif args.command == "bigquery":
        year, month = get_year_month()
        fetch_characteristics(args.credentials, year, month)

    elif args.command == "summary":
        year, month = get_year_month()
        print_summary_statistics(args.credentials, year, month)

    elif args.command == "webstats":
        print_web_stats(args.app_name)

    elif args.command == "all":
        year, month = get_year_month()
        fetch_characteristics(args.credentials, year, month)
        print_summary_statistics(args.credentials, year, month)
        print_web_stats()


def fetch_measure_tags(repo_path: Path) -> dict:
    """Load measure tag definitions from cloned repo"""
    tags_path = repo_path / "openprescribing" / "common" / "measure_tags.json"
    with open(tags_path) as f:
        return json.load(f)


def process_measures(repo_path: Path, tags_def: dict) -> tuple:
    """Analyze measures and return statistics"""
    measures_dir = repo_path / "openprescribing" / "measures" / "definitions"
    measure_files = list(measures_dir.glob("*.json"))

    tag_counts = defaultdict(int)
    reviewed_count = 0
    review_periods = []

    for measure_file in measure_files:
        with open(measure_file) as f:
            data = json.load(f)

        # Count tags
        for tag_key in data.get("tags", []):
            tag_name = tags_def.get(tag_key, {}).get("name", "Unknown Tag")
            tag_counts[tag_name] += 1

        # Track reviews
        if data.get("date_reviewed"):
            reviewed_count += 1
            if data.get("next_review"):
                date_reviewed = date.fromisoformat(data["date_reviewed"])
                next_review = date.fromisoformat(data["next_review"])
                review_periods.append((next_review - date_reviewed).days)

    return {
        "total_measures": len(measure_files),
        "tag_counts": dict(tag_counts),
        "reviewed_count": reviewed_count,
        "review_periods": review_periods,
    }


def print_measures_report(analysis: dict):
    """Format measures analysis as markdown report"""
    print("\n## Measures Analysis\n")

    # Basic counts
    print(f"Total measures: {analysis['total_measures']}")
    print(f"Measures ever reviewed: {analysis['reviewed_count']}\n")

    # Tag breakdown
    print("### By Tag\n")
    print("| Tag | Count |")
    print("|-----|-------|")
    for tag, count in sorted(analysis["tag_counts"].items(), key=lambda x: -x[1]):
        print(f"| {tag} | {count} |")

    # Review periods
    if analysis["review_periods"]:
        print("\n### Review Periods\n")
        print(
            f"Median review period: {statistics.median(analysis['review_periods'])} days"
        )
        print("\nReview period distribution:")
        print("| Days | Count |")
        print("|------|-------|")
        counts = defaultdict(int)
        for days in analysis["review_periods"]:
            counts[days] += 1
        for days, count in sorted(counts.items()):
            print(f"| {days} | {count} |")


def print_web_stats(app_names: list[str] = None):
    """Print website analytics from Plausible"""
    if app_names is None:
        app_names = ["measures", "price_per_unit", "ghost_generics", "concessions"]

    api_key = os.getenv("PLAUSIBLE_API_KEY")
    if not api_key:
        print("\nSkipping web stats - PLAUSIBLE_API_KEY not found in environment")
        return

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        # Common function to process metrics
        def process_metrics(stats, metrics_list):
            if not stats.get("results"):
                return {}
            metrics = stats["results"][0]["metrics"]
            return (
                dict(zip(metrics_list, metrics))
                if isinstance(metrics, list)
                else metrics
            )

        # Process each app
        for app_name in app_names:
            # Special handling for analyse app
            if app_name == "analyse":
                # Special handling for analyse app
                analyse_query = {
                    "site_id": "openprescribing.net",
                    "metrics": ["visitors", "visits", "pageviews"],
                    "dimensions": ["event:page"],
                    "date_range": "12mo",
                    "filters": [["contains", "event:page", ["analyse"]]],
                    "pagination": {"limit": 10000},
                }

                response = requests.post(
                    "https://plausible.io/api/v2/query",
                    headers=headers,
                    json=analyse_query,
                )
                response.raise_for_status()
                analyse_data = response.json()

                geography_stats = defaultdict(
                    lambda: {
                        "visitors": 0,
                        "visits": 0,
                        "pageviews": 0,
                        "bookmarks": set(),
                    }
                )
                total_stats = {"visitors": 0, "visits": 0, "pageviews": 0}
                bookmarked_stats = {"visitors": 0, "visits": 0, "pageviews": 0}

                for result in analyse_data.get("results", []):
                    page = result["dimensions"][0]
                    metrics = result["metrics"]

                    # Update totals for all analyse views
                    total_stats["visitors"] += metrics[0]
                    total_stats["visits"] += metrics[1]
                    total_stats["pageviews"] += metrics[2]

                    # Extract bookmarked searches
                    if "#" in page:
                        try:
                            _, hash_part = page.split("#", 1)
                            params = {}
                            for param in hash_part.split("&"):
                                if "=" in param:
                                    key, value = param.split("=", 1)
                                    params[key] = value
                            org = params.get("org", "").lower()

                            if org:
                                geography = org.split(",")[
                                    0
                                ]  # Take first org if multiple
                                geo_entry = geography_stats[geography]
                                geo_entry["visitors"] += metrics[0]
                                geo_entry["visits"] += metrics[1]
                                geo_entry["pageviews"] += metrics[2]
                                geo_entry["bookmarks"].add(hash_part)

                                # Update bookmarked totals
                                bookmarked_stats["visitors"] += metrics[0]
                                bookmarked_stats["visits"] += metrics[1]
                                bookmarked_stats["pageviews"] += metrics[2]

                        except Exception as e:
                            print(f"Error parsing {page}: {str(e)}")

                # Print analyse report
                print("\n## App: Analyse Traffic")
                print("Pages containing 'analyse' in their URL\n")
                print("### All Analyse Views (Last 12mo)")
                print(f"- Unique visitors: {total_stats['visitors']:,}")
                print(f"- Total visits: {total_stats['visits']:,}")
                print(f"- Page views: {total_stats['pageviews']:,}\n")

                print("### Bookmarked Searches")
                print(f"- Unique visitors: {bookmarked_stats['visitors']:,}")
                print(f"- Total visits: {bookmarked_stats['visits']:,}")
                print(f"- Page views: {bookmarked_stats['pageviews']:,}")
                print(
                    f"- Unique bookmarks: {sum(len(g['bookmarks']) for g in geography_stats.values()):,}\n"
                )

                print("#### Breakdown by Organisation")
                print(
                    "| Geography | Unique Visitors | % Visitors | Visits | Page Views | Unique orgs |"
                )
                print(
                    "|-----------|----------------:|-----------:|-------:|-----------:|------------------:|"
                )

                total_bookmarked_visitors = bookmarked_stats["visitors"]
                for geo in sorted(
                    geography_stats.items(), key=lambda x: -x[1]["visitors"]
                ):
                    name, stats = geo
                    percentage = (
                        (stats["visitors"] / total_bookmarked_visitors * 100)
                        if total_bookmarked_visitors
                        else 0
                    )

                    print(
                        f"| {name.title()} | "
                        f"{stats['visitors']:,} | "
                        f"{percentage:.1f}% | "
                        f"{stats['visits']:,} | "
                        f"{stats['pageviews']:,} | "
                        f"{len(stats['bookmarks']):,} |"
                    )

                continue  # Skip normal app processing

            # New approach: Get ALL raw data matching the URL pattern
            raw_query = {
                "site_id": "openprescribing.net",
                "metrics": ["visitors", "visits", "pageviews"],
                "dimensions": ["event:page"],
                "date_range": "12mo",
                "filters": [["matches", "event:page", [f"^/[^/]+/[^/]+/{app_name}/"]]],
                "pagination": {"limit": 10000},
            }

            response = requests.post(
                "https://plausible.io/api/v2/query", headers=headers, json=raw_query
            )
            response.raise_for_status()
            raw_data = response.json()

            # Process raw data for both existing and new reports
            geography_stats = defaultdict(
                lambda: {
                    "visitors": 0,
                    "visits": 0,
                    "pageviews": 0,
                    "unique_ids": set(),
                }
            )
            total_stats = {"visitors": 0, "visits": 0, "pageviews": 0}

            for result in raw_data.get("results", []):
                page = result["dimensions"][0]
                metrics = result["metrics"]

                # Extract geography and unique_id from URL
                match = re.match(
                    r"^/([^/]+)/([^/]+)/" + re.escape(app_name) + "/", page
                )
                if not match:
                    print(f"Warning: URL {page} doesn't match expected pattern")
                    continue

                geography, unique_id = match.groups()

                # Update geography stats
                geo_entry = geography_stats[geography]
                geo_entry["visitors"] += metrics[0]  # visitors
                geo_entry["visits"] += metrics[1]  # visits
                geo_entry["pageviews"] += metrics[2]  # pageviews
                geo_entry["unique_ids"].add(unique_id)

                # Update totals
                total_stats["visitors"] += metrics[0]
                total_stats["visits"] += metrics[1]
                total_stats["pageviews"] += metrics[2]

            # Print existing report using the same data
            print(f"\n## App: {app_name.title()} Traffic")
            print(f"Pages containing '/{app_name}/' in their URL path\n")
            print("### Last 12mo")
            print(f"- Unique visitors: {total_stats['visitors']:,}")
            print(f"- Total visits: {total_stats['visits']:,}")
            print(f"- Page views: {total_stats['pageviews']:,}")

            # New geography breakdown
            print("\n### Breakdown by Geography")
            print(
                "| Geography | Unique Visitors | % Visitors | Visits | Page Views | Unique orgs |"
            )
            print(
                "|-----------|----------------:|-----------:|-------:|-----------:|-----------:|"
            )

            # Sort by descending visitors
            for geo in sorted(geography_stats.items(), key=lambda x: -x[1]["visitors"]):
                name, stats = geo
                percentage = (
                    (stats["visitors"] / total_stats["visitors"] * 100)
                    if total_stats["visitors"]
                    else 0
                )

                print(
                    f"| {name.title()} | "
                    f"{stats['visitors']:,} | "
                    f"{percentage:.1f}% | "
                    f"{stats['visits']:,} | "
                    f"{stats['pageviews']:,} | "
                    f"{len(stats['unique_ids']):,} |"
                )

        # 2. Always show all pages report
        # Get earliest date
        date_response = requests.post(
            "https://plausible.io/api/v2/query",
            headers=headers,
            json={
                "site_id": "openprescribing.net",
                "metrics": ["visitors"],
                "dimensions": ["time:day"],
                "date_range": "all",
                "order_by": [["time:day", "asc"]],
                "pagination": {"limit": 1},
            },
        )
        date_response.raise_for_status()
        earliest_date = date_response.json()["results"][0]["dimensions"][0]

        # Get all pages metrics
        all_pages_stats = {}
        for period in ["12mo", "all"]:
            response = requests.post(
                "https://plausible.io/api/v2/query",
                headers=headers,
                json={
                    "site_id": "openprescribing.net",
                    "metrics": ["visitors", "visits", "pageviews"],
                    "date_range": period,
                },
            )
            response.raise_for_status()
            all_pages_stats[period] = process_metrics(
                response.json(), ["visitors", "visits", "pageviews"]
            )

        print("\n## All Website Traffic")
        print(f"Earliest data available: {earliest_date}\n")

        for period in ["12mo", "all"]:
            if stats := all_pages_stats.get(period):
                print(f"### Last 12 Months" if period == "12mo" else "### All Time")
                print(f"- Unique visitors: {stats.get('visitors', 0):,}")
                print(f"- Total visits: {stats.get('visits', 0):,}")
                print(f"- Page views: {stats.get('pageviews', 0):,}")

    except Exception as e:
        print(f"\nError fetching web stats: {str(e)}")


if __name__ == "__main__":
    main()
