# OpenPrescribing stats

This is a collection of scripts to generate counts of various things for our OpenPrescribing paper.

The output is in [`REPORT.md`](./REPORT.md).

## How to generate the report

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


# Citations metrics

It's basically impossible to get reasonably accurate citation metrics. Google Scholar is by far the most sensitive, and is known to have lots of accuracy issues [^1]. WoS is considered very specific, but misses out a lot of real papers.

A search on some of the most common websites using the keyword `openprescribing` gives the following:

| Source               | Access Type     | Count / Notes                                    |
| -------------------- | --------------- | ------------------------------------------------ |
| Scopus               | ❌ No full text | 150 papers, plus 26 preprints, 77 secondary docs |
| Web of Science (WoS) | ❌ No full text | 39 outputs                                       |
| CORE (core.ac.uk)    | ✅ Full text    | 165 outputs                                      |
| PubMed Central (PMC) | ✅ Full text    | 165 outputs                                      |
| BASE                 | ✅ Full text    | 57 outputs                                       |
| Google Scholar       | ✅ Full text    | \~850 results                                    |

The `citations.py` module in this repo uses PubMed _plus_ PubMed Central to get as much PubMed content as possible. The script fetches all papers matching 'openprescribing' from pubmed, where "matching" means "match _full text_ where available in PMC and _metadata_ from PM where not". However, using pubmed means we will not pick up:

- papers not covered by pubmed (mainly, this means papers not in biosciences)
- non-open-access papers which mention openprescribing outside metadata (for example, only in references)

I compared the output of this PubMed search with a Google Scholar search, by using [Publish or Perish](https://harzing.com/resources/publish-or-perish) with a keyword search of `"openprescribing"` (quotes are required). The results are in `op_papers.csv`.

There are 672 in Google Scholar, compared with 142 using the Pubmed method.

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

Next, I wrote [a script](./gs_analysis.py) to use an LLM to classify the Google Scholar matches:

- Download the article URL that Google Scholar provides (where available)
- Of the ones which it's possible to download:
  - Classify them with a prompt, in order to work out which ones are real academic papers
  - Also classify them as the entire article, or abstract only
  - For those which are the entire article, report on the ones which contain the string "opensafely"

The [analysis report](./gs_analysis_report.md) shows:

- 588 non-citation results from Google
- Of these, 369 had content that could be downloaded
- Of these, 174 were classified as academic papers (the rest were preprints, briefings, editorials, theses, etc)
- Of these, 144 were classfied as the full paper, rather than just the abstract
- Of these, 108 (75%) contained the string "openprescribing"

If we assume

- 25% of all hits for openprescribing in Google Scholar are simply wrong
- 53% of all hits in Google Scholar are for non-paper types

Then this implies around 207 "true" papers in the Google Scholar corpus.

[^1]: Romy Sauvayre, [Types of Errors Hiding in Google Scholar Data](https://www.jmir.org/2022/5/e28354/), JMIR 2022
