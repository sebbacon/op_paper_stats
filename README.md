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

Here's a random sample of papers that were in the google scholar report, but not the pubmed. If this is representative seems like approximately 35% over ascertainment.

- 4 of them are definitely legit papers.
- 3 are legit but very short paper-lite
- 4 were definitely not papers
- 1 was a preprint

1. [Hyponatraemia in primary care(https://www.bmj.com/content/365/bmj.l1774.abstract). A "Practice Pointer" in BMJ, rather than a paper. Recorded as "cited by 10" in Google Scholar
2. [Inequalities in prescription rates of anti-osteoporosis drugs in primary care in England: A practice-level prescribing data analysis in 2013-2018](https://www.sciencedirect.com/science/article/abs/pii/S8756328219304193). Possibly because not open access?
3. [Investigating Impacts of Health Policies Using Staggered Difference-inDifferences: The Effects of Adoption of an Online Consultation System
   on Prescribing Patterns of Antibiotics](https://arxiv.org/pdf/2305.19878) - a preprint. Cited by 1
4. [North East Lincolnshire MENTAL HEALTH AND WELL BEING NEEDS ASSESSMENT 2018](https://www.nelincsdata.net/wp-content/uploads/NELC_Mental_Health_and_Wellbeing_Report_2019.pdf) - not a paper
5. [OpenPrescribing: normalised data and software tool to research trends in English NHS primary care prescribing 1998–2016. BMJ Open. 2018; 8 (2) ](<https://scholar.google.com/scholar?q=OpenPrescribing%3A%20normalised%20data%20and%20software%20tool%20to%20research%20trends%20in%20English%20NHS%20primary%20care%20prescribing%201998%E2%80%932016.%20BMJ%20Open.%202018%3B%208%20(2)%20%E2%80%A6>) -- apears as a "citation" as well as the paper - this is the former
6. [P38 ‘Getting it right first time’ (GIRFT) in the management of COPD](https://www.proquest.com/openview/bc2f6ddbbd8c31e480399e7109b8fb45/1?pq-origsite=gscholar&cbl=2041050) - a poster, not a paper (there were 5 of these from one conference)
7. [Pharmacist intervention in cardiovascular disease prevention: lipid modifying treatment optimisation in type 2 diabetes within hastings primary care network](https://academic.oup.com/ijpp/article/30/Supplement_2/ii27/6854518) - a proper academic article, but not really a paper (very short). Probably valid
8. [Physical ability of people with rheumatoid arthritis and age-sex matched controls to use four commonly prescribed inhaler devices](https://www.sciencedirect.com/science/article/pii/S0954611117304328) - "short communication" article type. [Was in pubmed](https://pubmed.ncbi.nlm.nih.gov/29414447/), so unclear why not found in pubmed search.
9. [Real-world effectiveness of oral anticoagulants in the prevention of stroke: emulation and extension of the ARISTOTLE trial using UK EHRs](https://researchonline.lshtm.ac.uk/id/eprint/4674716/1/2024_EPH_PhD_Teoh_M.pdf) - this is a doctoral thesis
10. [Risk Elements Linked to Oral Candidiasis](file:///Users/sebbacon/Downloads/79325_93-98.pdf) - looks like a legit paper, not in Pubmed
11. [Social Media and Open Data to Quantify the Effects of Noise on Health](https://www.frontiersin.org/journals/sustainable-cities/articles/10.3389/frsc.2020.00041/full) - legit paper, not in pubmed
12. [Tesco Grocery 1.0, a large-scale dataset of grocery purchases in London](https://www.nature.com/articles/s41597-020-0397-7) - legit paper

[^1]: Romy Sauvayre, [Types of Errors Hiding in Google Scholar Data](https://www.jmir.org/2022/5/e28354/), JMIR 2022
