import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path


def load_and_match_data(
    file_1,
    file_2,
    sheet_1=0,
    match_column="test",
    value_column_1="best",
    value_column_2="score"
):
    df_1 = pd.read_excel(file_1, sheet_name=sheet_1, engine = "calamine")
    df_2 = pd.read_csv(file_2)

    # Clean the matching column
    df_1[match_column] = df_1[match_column].astype(str).str.strip()
    df_2[match_column] = df_2[match_column].astype(str).str.strip()

    # Keep only required columns
    df_1 = df_1[[match_column, value_column_1]].copy()
    df_2 = df_2[[match_column, value_column_2]].copy()

    # Rename value columns to consistent internal names
    df_1 = df_1.rename(
        columns={value_column_1: "value_file_1"}
    )

    df_2 = df_2.rename(
        columns={value_column_2: "value_file_2"}
    )

    # Match rows using the match column
    matched = pd.merge(
        df_1,
        df_2,
        on=match_column,
        how="inner"
    )

    # Convert values to numeric
    matched["value_file_1"] = pd.to_numeric(
        matched["value_file_1"],
        errors="coerce"
    )

    matched["value_file_2"] = pd.to_numeric(
        matched["value_file_2"],
        errors="coerce"
    )

    # Remove rows with missing values
    matched = matched.dropna(
        subset=["value_file_1", "value_file_2"]
    )

    return matched


def calculate_pearson(matched):
    if len(matched) < 2:
        raise ValueError(
            "At least two matched observations are required."
        )

    corr, p_value = pearsonr(
        matched["value_file_1"],
        matched["value_file_2"]
    )

    return corr, p_value


def create_scatter_plot(
    matched,
    corr,
    p_value,
    file_1,
    file_2,
    value_column_1,
    value_column_2,
    output_file="pearson_scatter_plot.png"
):
    plt.figure(figsize=(8, 6))

    plt.scatter(
        matched["value_file_1"],
        matched["value_file_2"],
        alpha=0.7
    )

    plt.xlabel(
        f"{Path(file_1).name} - {value_column_1}"
    )

    plt.ylabel(
        f"{Path(file_2).name} - {value_column_2}"
    )

    plt.title(
        f"Pearson correlation\n"
        f"r = {corr:.3f}, "
        f"p = {p_value:.3g}, "
        f"n = {len(matched)}"
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def main():

    # -----------------------------
    # Settings
    # -----------------------------
    biopsies = ["Func050", "Func116"]
    for biopsy in biopsies:
        file_1 = f"/media/jenny/Expansion/jenny_funcprost/conic/results/New/Fra_Vilde/Quantification/Data/{biopsy}_Conic_SpotQuantification.xlsx"
        file_2 = f"/media/jenny/Expansion/jenny_funcprost/pannuke/results/zhang_original_weights/New/quantification_jenny/data/{biopsy}_Conic_SpotQuantification.csv"

        sheet_1 = 0

        match_column = "test"

        # These can have different names
        value_column_1 = "best"
        value_column_2 = "result"

        # -----------------------------
        # Load and match data
        # -----------------------------

        matched = load_and_match_data(
            file_1=file_1,
            file_2=file_2,
            sheet_1=sheet_1,
            match_column=match_column,
            value_column_1=value_column_1,
            value_column_2=value_column_2
        )

        print("\nMatched data:")
        print(matched)

        print(
            f"\nNumber of matched observations: {len(matched)}"
        )

        # -----------------------------
        # Pearson correlation
        # -----------------------------

        corr, p_value = calculate_pearson(matched)

        print("\nPearson correlation results")
        print("---------------------------")
        print(f"Pearson correlation (r): {corr:.6f}")
        print(f"P-value: {p_value:.6g}")

        if p_value < 0.05:
            print(
                "Statistically significant at alpha = 0.05."
            )
        else:
            print(
                "Not statistically significant at alpha = 0.05."
            )

        # -----------------------------
        # Scatter plot
        # -----------------------------

        create_scatter_plot(
            matched=matched,
            corr=corr,
            p_value=p_value,
            file_1=file_1,
            file_2=file_2,
            value_column_1=value_column_1,
            value_column_2=value_column_2,
            output_file=f"/media/jenny/Expansion/jenny_funcprost/pannuke/results/zhang_original_weights/New/quantification_jenny/pearson/{biopsy}_pearson_scatter_plot.png"
        )

        # Save matched values
        matched.to_excel(
            f"/media/jenny/Expansion/jenny_funcprost/pannuke/results/zhang_original_weights/New/quantification_jenny/pearson/{biopsy}matched_data.xlsx",
            index=False,
            engine = "calamine"
        )


if __name__ == "__main__":
    main()