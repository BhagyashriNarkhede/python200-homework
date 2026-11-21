
# project_07.py
# Week 7 Mini-Project: World Happiness Agent

from pathlib import Path
import os
os.environ["MPLBACKEND"] = "Agg"

import pandas as pd
from scipy.stats import pearsonr
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, tool


# --------------------------------------------------
# 1. Environment Setup
# --------------------------------------------------

# Load the OpenAI API key from the repository root.
load_dotenv("../.env")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found. Check the root .env file."
    )

# Define the data and output directories.
DATA_DIR = Path("happiness_project")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared DataFrame used by the tools.
df = None


# --------------------------------------------------
# 2. Tool 1: Load Happiness Data
# --------------------------------------------------

@tool
def load_happiness_data() -> dict:
    """Load and combine all yearly World Happiness CSV files.

    Reads the yearly CSV files, normalizes their column names,
    extracts the year from each filename, and combines the data.

    Returns:
        A dictionary containing the DataFrame shape and column names.
    """
    global df

    column_names = {
        "Country": "country",
        "Region": "region",
        "Regional indicator": "region",
        "Happiness score": "happiness_score",
        "Ladder score": "happiness_score",
        "GDP per capita": "gdp_per_capita",
    }

    frames = []

    try:
        for path in sorted(
            DATA_DIR.glob("world_happiness_*.csv")
        ):
            year_df = pd.read_csv(
                path,
                sep=";",
                decimal=","
            )

            year_df = year_df.rename(
                columns=column_names
            )

            year_df["year"] = int(
                path.stem.split("_")[-1]
            )

            frames.append(year_df)

        if not frames:
            return {
                "error": "No World Happiness CSV files were found."
            }

        df = pd.concat(
            frames,
            ignore_index=True
        )

        return {
            "shape": df.shape,
            "columns": df.columns.tolist()
        }

    except Exception as error:
        return {"error": str(error)}


# --------------------------------------------------
# 3. Tool 2: Summarize a Column
# --------------------------------------------------

@tool
def summarize_column(column: str) -> dict:
    """Calculate descriptive statistics for a dataset column.

    Args:
        column: The name of the column to summarize.

    Returns:
        A dictionary containing descriptive statistics,
        or an error message if the column is unavailable.
    """
    if df is None:
        return {
            "error": "Please load the happiness data first."
        }

    if column not in df.columns:
        return {
            "error": f"Column '{column}' was not found."
        }

    try:
        return df[column].describe().to_dict()

    except Exception as error:
        return {"error": str(error)}


# --------------------------------------------------
# 4. Tool 3: Compute Correlation
# --------------------------------------------------

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Calculate the Pearson correlation between two columns.

    Args:
        col1: The name of the first numeric column.
        col2: The name of the second numeric column.

    Returns:
        A dictionary containing the column names,
        Pearson correlation coefficient, and p-value.
    """
    if df is None:
        return {
            "error": "Please load the happiness data first."
        }

    if col1 not in df.columns:
        return {
            "error": f"Column '{col1}' was not found."
        }

    if col2 not in df.columns:
        return {
            "error": f"Column '{col2}' was not found."
        }

    try:
        valid_data = df[[col1, col2]].dropna()

        pearson_r, p_value = pearsonr(
            valid_data[col1],
            valid_data[col2]
        )

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(pearson_r), 4),
            "p_value": round(float(p_value), 4),
        }

    except Exception as error:
        return {"error": str(error)}


# --------------------------------------------------
# 5. Tool 4: Get Top N Countries
# --------------------------------------------------

@tool
def get_top_n_countries(
    column: str,
    year: int,
    n: int = 5
) -> dict:
    """Find the top N countries for a selected column and year.

    Args:
        column: The numeric column used to rank countries.
        year: The year to filter the dataset by.
        n: Number of top countries to return. Defaults to 5.

    Returns:
        A dictionary containing the year, column, and
        a list of the top countries.
    """
    if df is None:
        return {
            "error": "Please load the happiness data first."
        }

    if column not in df.columns:
        return {
            "error": f"Column '{column}' was not found."
        }

    if "country" not in df.columns or "year" not in df.columns:
        return {
            "error": "Required country or year columns are missing."
        }

    if n < 1:
        return {
            "error": "n must be at least 1."
        }

    try:
        year_data = df[df["year"] == year]

        if year_data.empty:
            return {
                "error": f"No data found for year {year}."
            }

        top_countries = (
            year_data[["country", column]]
            .dropna(subset=[column])
            .sort_values(
                by=column,
                ascending=False
            )
            .head(n)
        )

        return {
            "year": year,
            "column": column,
            "top_countries": top_countries.to_dict(
                orient="records"
            ),
        }

    except Exception as error:
        return {"error": str(error)}


# --------------------------------------------------
# 6. Create the World Happiness CodeAgent
# --------------------------------------------------

def create_happiness_agent():
    """Create and configure the World Happiness CodeAgent.

    Returns:
        A configured CodeAgent with the four happiness tools.
    """

    model = OpenAIServerModel(
        api_key=api_key,
        model_id="gpt-4o-mini"
    )
    
    system_prompt = r"""
    You are a World Happiness data analysis assistant.

    You help users explore and understand the World Happiness dataset.

    You have access to these four tools:

    1. load_happiness_data
    2. summarize_column
    3. compute_correlation
    4. get_top_n_countries

    Use load_happiness_data before using the other tools.

    Use the tools whenever they can answer the user's question.

    IMPORTANT:
    The load_happiness_data tool returns only metadata:
    the dataset shape and column names. It does NOT return
    the actual DataFrame.

    For plotting tasks, load the actual CSV files directly
    using pandas.

    The CSV files are stored in the happiness_project directory.

    Their filenames follow this pattern:
    world_happiness_2015.csv through world_happiness_2024.csv.

    The CSV files use semicolon separators and decimal commas.

    When loading the CSV files:

    1. Use pathlib.Path to find the files with:
       Path("happiness_project").glob("world_happiness_*.csv")

    2. Do not assume filenames are simply 2015.csv or 2016.csv.

    3. Read each CSV using:
       pd.read_csv(file, sep=";", decimal=",")

    4. Extract the four-digit year from each filename using
       a regular expression such as r"(\d{4})".

    5. Rename the columns using this mapping:
       Country -> country
       Region -> region
       Regional indicator -> region
       Happiness score -> happiness_score
       Ladder score -> happiness_score
       GDP per capita -> gdp_per_capita

    6. Add the extracted year as an integer year column.

    7. Combine all yearly DataFrames using pd.concat().

    8. Use the actual CSV records for all analysis and plots.

    NEVER create mock, fake, random, sample, or invented data.

    NEVER use the metadata dictionary returned by
    load_happiness_data() as a DataFrame.

    If the CSV files cannot be loaded, explain the problem
    instead of creating a plot with fabricated data.

    For general plotting tasks:
    - Use pandas and matplotlib.pyplot.
    - Use the actual CSV data.
    - Choose an appropriate chart type.
    - Save requested plots in the outputs directory.

    For happiness-by-region line charts, follow these steps:

    1. Load all CSV files from happiness_project.
    2. Remove rows with missing region or happiness_score.
    3. Group the data by year and region.
    4. Calculate the mean happiness_score for each year
       and region.
    5. Create a line chart with year on the x-axis.
    6. Use average happiness_score on the y-axis.
    7. Plot one separate line for each region.
    8. Do not plot individual country records as connected lines.
    9. Do not include missing region names or a region named nan.
    10. Include all available years, including 2024.
    11. Add a chart title, axis labels, and a legend.
    12. Save the chart to:
        outputs/happiness_by_region.png

    For other plotting tasks:
    - Use the actual CSV data.
    - Create the appropriate chart requested by the user.
    - Save the plot in the outputs directory.

    Create the outputs directory if it does not exist.

    Before importing matplotlib.pyplot, set the Matplotlib
    backend to Agg using matplotlib.use("Agg").

    Never use plt.show().
    Save plots using plt.savefig().
    Close plots using plt.close().

    Explain your findings clearly and accurately.
    Do not claim that correlation proves causation.
    """    


    agent = CodeAgent(
        tools=[
            load_happiness_data,
            summarize_column,
            compute_correlation,
            get_top_n_countries,
        ],
        model=model,
        additional_authorized_imports=[
            "pandas",
            "matplotlib",
            "matplotlib.pyplot",
            "scipy.stats",
            "pathlib",
            "re",
        ],
        max_steps=8,
        instructions=system_prompt,
    )

    return agent

# --------------------------------------------------
# 7. Run the Guided Queries
# --------------------------------------------------

if __name__ == "__main__":

    print("\n======================================")
    print("WORLD HAPPINESS AGENT")
    print("======================================")

    agent = create_happiness_agent()

    # Query 1: Load the dataset.
    query_1 = (
        "Load the happiness data and tell me its shape "
        "and column names."
    )

    print("\n--- Query 1: Load Happiness Data ---")
    print(agent.run(query_1, reset=False))

    # Query 2: Summarize happiness scores.
    query_2 = (
        "Summarize the happiness_score column."
    )

    print("\n--- Query 2: Happiness Score Summary ---")
    print(agent.run(query_2, reset=False))

    # Query 3: Calculate GDP and happiness correlation.
    query_3 = (
        "What is the correlation between gdp_per_capita "
        "and happiness_score? Is it statistically significant?"
    )

    print("\n--- Query 3: GDP and Happiness Correlation ---")
    print(agent.run(query_3, reset=False))

    # Query 4: Find the top five happiest countries in 2020.
    query_4 = (
        "Show me the top 5 happiest countries in 2020."
    )

    print("\n--- Query 4: Top 5 Countries in 2020 ---")
    print(agent.run(query_4, reset=False))

    # Query 5: Create a line chart by region.
    query_5 = (
        "Load the actual yearly CSV files. Calculate the average "
        "happiness_score for each region in each year from 2015 "
        "through 2024. Plot these averages as a line chart with "
        "one separate line per region. Put year on the x-axis "
        "and average happiness score on the y-axis. "
        "Save the plot to outputs/happiness_by_region.png."
    )

    print("\n--- Query 5: Happiness by Region Plot ---")
    print(agent.run(query_5, reset=False))

    # --------------------------------------------------
    # 8. Your Own Queries
    # --------------------------------------------------

    # Own Query 1: This query should use get_top_n_countries.
    own_query_1 = (
        "Which are the top 5 countries with the highest "
        "GDP per capita in 2022?"
    )

    print("\n--- Own Query 1: Top GDP Countries ---")
    print(agent.run(own_query_1, reset=False))

    # Own Query 2: This query requires generated Python code
    # to analyze and plot the data.
    own_query_2 = (
        "Create a bar chart showing the average happiness score "
        "for each region in 2020. Save it as "
        "outputs/average_happiness_by_region_2020.png. "
        "Use pandas and matplotlib. Do not display the plot."
    )

    print("\n--- Own Query 2: Regional Happiness Bar Chart ---")
    print(agent.run(own_query_2, reset=False))


# --------------------------------------------------
# 9. Reflection
# --------------------------------------------------

# # --- Reflection ---
#
# 1. In Query 3, the agent reported a Pearson correlation coefficient (r) of
#    0.6313 and a p-value of 0.0. It described the relationship as a strong
#    positive correlation and noted that it was statistically significant.
#    This was a sensible interpretation because the positive value indicates
#    that higher GDP per capita tends to be associated with higher happiness
#    scores. The p-value was rounded to four decimal places, so 0.0 does not
#    mean the actual p-value is exactly zero. Overall, the agent communicated
#    the direction, strength, and statistical significance appropriately.
#
# 2. I was surprised by the agent's ability to recover from errors while
#    creating the happiness-by-region line chart. Initially, it failed because
#    the dataset used the column name "Regional indicator" instead of "Region".
#    After checking the CSV column names, the agent corrected the column mapping,
#    calculated the average happiness scores for each region and year, and
#    successfully created the chart for 2015 through 2024. This showed that
#    the agent could use tools and adjust its approach when it encountered
#    problems, although it needed several attempts.
#
# 3. An additional tool I would find useful is a compare_countries tool.
#    It would accept two or more country names and return their happiness
#    scores across the available years. This would help the agent answer
#    questions such as "How have Finland and the United States' happiness
#    scores changed from 2015 to 2024?" It could make comparisons over time
#    easier and provide more detailed insights into country-level trends.