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

I wrote a script (with Claude Code) to help visualise the differences - [here's the output](https://sebbacon.github.io/op_paper_stats/paper_comparison_report.html). (You can regenerate it against fresh data with `run_analysis.sh`)

I took a random sample of 12 papers that were in the google scholar report, but not the pubmed.

| #   | Title (linked)                                                                                                                                                                                                                                                                      | Category            | Notes                         |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ----------------------------- |
| 1   | [Hyponatraemia in primary care](https://www.bmj.com/content/365/bmj.l1774.abstract)                                                                                                                                                                                                 | ❌ Not a paper      | BMJ "Practice Pointer", cited |
| 2   | [Inequalities in prescription rates...](https://www.sciencedirect.com/science/article/abs/pii/S8756328219304193)                                                                                                                                                                    | ✅ Legit paper      | Possibly missed due to access |
| 3   | [Impacts of Health Policies (arXiv)](https://arxiv.org/pdf/2305.19878)                                                                                                                                                                                                              | 🧪 Preprint         | Cited by 1                    |
| 4   | [NEL Mental Health Needs](https://www.nelincsdata.net/wp-content/uploads/NELC_Mental_Health_and_Wellbeing_Report_2019.pdf)                                                                                                                                                          | ❌ Not a paper      | Public health report          |
| 5   | [OpenPrescribing citation](https://scholar.google.com/scholar?q=OpenPrescribing%3A%20normalised%20data%20and%20software%20tool%20to%20research%20trends%20in%20English%20NHS%20primary%20care%20prescribing%201998%E2%80%932016.%20BMJ%20Open.%202018%3B%208%20%282%29%20%E2%80%A6) | ❌ Not a paper      | Citation reference only       |
| 6   | [GIRFT COPD (poster)](https://www.proquest.com/openview/bc2f6ddbbd8c31e480399e7109b8fb45/1?pq-origsite=gscholar&cbl=2041050)                                                                                                                                                        | ❌ Not a paper      | Conference poster             |
| 7   | [Pharmacist intervention](https://academic.oup.com/ijpp/article/30/Supplement_2/ii27/6854518)                                                                                                                                                                                       | 🪶 Short paper-lite | Very brief, borderline        |
| 8   | [RA inhaler study](https://www.sciencedirect.com/science/article/pii/S0954611117304328)                                                                                                                                                                                             | 🪶 Short paper-lite | “Short communication”         |
| 9   | [Oral anticoagulants thesis](https://researchonline.lshtm.ac.uk/id/eprint/4674716/1/2024_EPH_PhD_Teoh_M.pdf)                                                                                                                                                                        | ❌ Not a paper      | PhD thesis                    |
| 10  | [Oral Candidiasis](file:///Users/sebbacon/Downloads/79325_93-98.pdf)                                                                                                                                                                                                                | ✅ Legit paper      | Not in PubMed                 |
| 11  | [Noise & Health (Frontiers)](https://www.frontiersin.org/journals/sustainable-cities/articles/10.3389/frsc.2020.00041/full)                                                                                                                                                         | ✅ Legit paper      | Not in PubMed                 |
| 12  | [Tesco Grocery Dataset](https://www.nature.com/articles/s41597-020-0397-7)                                                                                                                                                                                                          | ✅ Legit paper      | Not in PubMed                 |

### Summary:

- ✅ Legit papers: 4
- 🪶 Short paper-lite: 3
- ❌ Not papers: 4
- 🧪 Preprint: 1

Approx. 35% over-ascertainment in Google Scholar vs PubMed appears plausible.

[^1]: Romy Sauvayre, [Types of Errors Hiding in Google Scholar Data](https://www.jmir.org/2022/5/e28354/), JMIR 2022
