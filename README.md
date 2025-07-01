# TL;DR

To set up:

- Obtain a Plausible API key and add it as PLAUSIBLE_API_KEY to .env
- Obtain GCP credentials with read access to bigquery for the embdatalab project, and add to gcp-credentials.json
- Download data from here: https://openprescribing.net/admin/auth/user/exports/alert-signups.csv and put it in data/

Then, to generate report:

    python citations.py fetch --output papers.csv
    # edit papers.csv: add a core_papers column with `1` to mark papers that you consider core to the project,
    # and `0` for papers that you don't (these are potentially citing papers)
    python citations.py metrics --input papers.csv > REPORT.md
    echo >> REPORT.md
    python alert_signups.py >> REPORT.md
    echo >> REPORT.md
    python code_stats.py >> REPORT.md
    echo >> REPORT.md
    python characteristics.py all >> REPORT.md

# Citations metrics

The `citations.py` module uses PubMed to attempt to get citation info.

It fetches all papers matching 'openprescribing' from pubmed, where "matching" means "full text where available and metadata where not".

It uses pubmed because Google Scholar is known to have lots of accuracy issues [^1]

However, using pubmed means we will not pick up:

- papers not covered by pubmed (mainly, this means papers not in biosciences)
- non-open-access papers which mention openprescribing outside metadata (for example, only in references)

To get paper metrics and write to a CSV:

    python data/citations/fetch.py metrics --output <path>

Now you'll need to add a core_papers column with `1` to mark papers that you consider core to the project, and `0` for papers that you don't -- the idea is that these are things that cite the core papers.

Now, to print metrics:

    python data/citations/fetch.py metrics --input <path>

I compared this with a Google Scholar search, by using [Publish or Perish](https://harzing.com/resources/publish-or-perish) with a keyword search of `openprescribing`. The results are in `op_papers.csv`. There are 672, compared with 142 using the Pubmed method.

I wrote a script (with Claude Code) to help visualise the differences - [here's the output]().

Regenerate it with `run_analysis.sh`

[^1]: Romy Sauvayre, [Types of Errors Hiding in Google Scholar Data](https://www.jmir.org/2022/5/e28354/), JMIR 2022
