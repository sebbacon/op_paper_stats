# TL;DR

To set up:

- Obtain a Plausible API key and add it as PLAUSIBLE_API_KEY to .env
- Obtain GCP credentials with read access to bigquery for the embdatalab project, and add to gcp-credentials.json
- Download data from here: https://openprescribing.net/admin/auth/user/exports/alert-signups.csv and put it in data/

To generate report:

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

This function fetches all papers matching 'openprescribing' from pubmed, where "matching" means "full text where available and metadata where not".

It uses pubmed because Google Scholar is known to have lots of accuracy issues [^1]

This means we will not pick up:

- papers not covered by pubmed (mainly, this means papers not in biosciences)
- non-open-access papers which mention openprescribing outside metadata (for example, only in references)

To get paper metrics and write to a CSV:

    python data/citations/fetch.py metrics --output <path>

Now you'll need to add a core_papers column with `1` to mark papers that you consider core to the project, and `0` for papers that you don't -- the idea is that these are things that cite the core papers.

Now, to print metrics:

    python data/citations/fetch.py metrics --input <path>

Finally, in case you want it in an appendix, to print a list of the core papers:

     python data/citations/fetch.py list-core --input <path>

[^1]: Romy Sauvayre, [Types of Errors Hiding in Google Scholar Data](https://www.jmir.org/2022/5/e28354/), JMIR 2022

_Double check this against GS_

> I think it is legitimate to cite scholar "openprescribing.net" search (n=571) for what I think you might be writing as long as we report all the numbers with some explanatory text
