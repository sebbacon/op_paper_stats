import pandas as pd
import os


def load_alert_data():
    """Load and preprocess alert signups data"""
    df = pd.read_csv(os.path.join("data", "alert-signups.csv"))
    df["created_at"] = pd.to_datetime(df["created_at"])

    # Simplified org_type assignment for practices
    df["org_type"] = None
    df.loc[:, "org_type"] = df.loc[:, "org_id"].apply(
        lambda x: (
            "practice"
            if isinstance(x, str) and x[:1].isalpha() and x[1:].isdigit()
            else "other"
        )
    )

    # Clean alert type labels
    alert_type_map = {
        "monthly": "Monthly alerts",
        "price_concessions": "Price Concession alerts",
        "search": "Analyse page alerts",
    }
    df["alert_type"] = df["alert_type"].replace(alert_type_map)

    return df


def add_totals(df, index_col, total_label):
    """Add totals row to dataframe"""
    totals = df.select_dtypes(include="number").sum().to_frame().T
    totals[index_col] = f"<b>{total_label}</b>"
    return pd.concat([df, totals], ignore_index=True)


def print_markdown_table(title, df):
    """Print dataframe as markdown table with title"""
    print(f"### {title}\n")
    print(df.to_markdown(index=False, tablefmt="github"))
    print("\n")


def main():
    df = load_alert_data()

    # Table 1: Signups by alert type
    alert_type_counts = df.groupby("alert_type").size().reset_index(name="count")
    alert_type_table = add_totals(alert_type_counts, "alert_type", "Total")
    alert_type_table.columns = ["Alert Type", "Number of Signups"]
    print("# Alert email signups")
    print_markdown_table("## Number of signups by alert type", alert_type_table)

    # Table 2: Practices with at least one signup
    practice_signups = df[df["org_type"] == "practice"]["org_name"].nunique()
    practice_table = pd.DataFrame(
        {"Metric": ["Practices with at least one signup"], "Count": [practice_signups]}
    )
    print_markdown_table("## Practice engagement", practice_table)


if __name__ == "__main__":
    main()
