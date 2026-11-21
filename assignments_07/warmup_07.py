# Week 7 Assignment
# Part 1: Warmup Exercises

from pathlib import Path
import os
import json
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from dotenv import load_dotenv
from openai import OpenAI

from smolagents import (
    tool,
    ToolCallingAgent,
    CodeAgent,
    OpenAIServerModel,
)


# ============================================================
# Environment Setup
# ============================================================

# Load the OpenAI API key from the repository root .env file.
load_dotenv("../.env")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found. Check your repository root .env file."
    )

client = OpenAI(api_key=api_key)


# ============================================================
# Q1: Celsius to Fahrenheit Tool
# ============================================================

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit."""

    # Formula: Fahrenheit = (Celsius * 9/5) + 32
    fahrenheit = (celsius * 9 / 5) + 32

    return f"{celsius}°C is {fahrenheit}°F"


# JSON schema describing the function to an LLM.
celsius_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "The temperature in degrees Celsius."
                }
            },
            "required": ["celsius"],
            "additionalProperties": False
        },
        "strict": True
    }
}


# Test the function directly with three temperatures.
print("\n--- Q1: Celsius to Fahrenheit ---")

print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


# ============================================================
# Q2: Basic ReAct Agent using get_current_time
# ============================================================

# Define the get_current_time tool.
def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# JSON schema for the get_current_time tool.
time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current date and time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    }
}


def run_agent(user_query):
    """
    Run a basic agent with only the get_current_time tool.
    """

    messages = [
        {
            "role": "user",
            "content": user_query
        }
    ]

    # Q2 agent has only the time tool.
    tools = [time_schema]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # Check whether the model requested a tool.
    if message.tool_calls:
        messages.append(message.model_dump())

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(
                tool_call.function.arguments or "{}"
            )

            if tool_name == "get_current_time":
                result = get_current_time(**tool_args)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str)
            })

        # Ask the model to respond using the tool result.
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )

        return second_response.choices[0].message.content

    return message.content


# Q2 test
print("\n--- Question 2 ---")

# Prediction: The agent should answer directly without calling a tool,
# because Celsius-to-Fahrenheit conversion does not require the time tool.

response = run_agent("Convert 100 degrees Celsius to Fahrenheit.")

print("Agent response:", response)


# ============================================================
# Q3: Extend the Agent with Celsius Conversion
# ============================================================

def run_agent_with_celsius(user_query):
    """
    Run an agent with both the time and Celsius conversion tools.
    """

    messages = [
        {
            "role": "user",
            "content": user_query
        }
    ]

    # Both tools are now available to the model.
    tools = [time_schema, celsius_schema]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # Execute requested tools.
    if message.tool_calls:
        messages.append(message.model_dump())

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(
                tool_call.function.arguments or "{}"
            )

            if tool_name == "get_current_time":
                result = get_current_time(**tool_args)

            elif tool_name == celsius_schema["function"]["name"]:
                celsius_function = globals()[tool_name]
                result = celsius_function(**tool_args)

            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str)
            })

        # Ask the model to respond using the tool result.
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )

        return second_response.choices[0].message.content

    return message.content


# Q3 Test A: Celsius conversion
print("\n--- Question 3A ---")

response_a = run_agent_with_celsius(
    "Convert 37 degrees Celsius to Fahrenheit."
)

print("Agent response:", response_a)


# Q3 Test B: Plain-English question
print("\n--- Question 3B ---")

response_b = run_agent_with_celsius(
    "What is the boiling point of water in Celsius?"
)

print("Agent response:", response_b)


# Q2 prediction:
#
# Will run_agent("Convert 100 degrees Celsius to Fahrenheit")
# trigger a tool call?
#
# Prediction: No. The agent only has get_current_time as a tool.
# It does not have a Celsius conversion tool, so the model can
# answer the conversion question directly.
#
# How many API calls will be made?
#
# Prediction: One API call, because the model can answer directly.
#
# Q3:
# The agent now has both the time tool and Celsius conversion tool.
# The model can choose whether to call a tool based on the question.


# ============================================================
# Q4: CSV Manager Class
# ============================================================

class CsvManager:

    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"

        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []

        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"

            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }

        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """

        files = self._available_csv_files()

        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }

        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """

        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """

        error = self._ensure_loaded()

        if error:
            return error

        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """

        error = self._ensure_loaded()

        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [
                column for column in columns
                if column not in self.df.columns
            ]

            if missing:
                return {
                    "error": f"These columns are not in the data: {missing}"
                }

            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)

        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """

        error = self._ensure_loaded()

        if error:
            return error

        if column not in self.df.columns:
            return {
                "error": (
                    f"'{column}' is not a column. "
                    f"Options: {self.df.columns.tolist()}"
                )
            }

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}

        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(
        self,
        y: str,
        x: str | None = None,
        plot_type: str = "line"
    ):
        """
        Plot from the active CSV.

        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """

        error = self._ensure_loaded()

        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return (
                f"Error: column '{y}' "
                f"is not in {self.df.columns.tolist()}"
            )

        # If someone accidentally passes x == y, treat it like "plot y".
        if x == y:
            x = None

        # Scatter needs x.
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"

        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(
                f"{title_csv} | Line plot: {y} vs row index"
            )

            plt.show()

            return f"Plotted {y} vs row index as a line plot."

        if x not in self.df.columns:
            return (
                f"Error: column '{x}' "
                f"is not in {self.df.columns.tolist()}"
            )

        ax = self.df.plot(
            x=x,
            y=y,
            kind=plot_type
        )

        ax.set_title(
            f"{title_csv} | {plot_type.title()} plot: {y} vs {x}"
        )

        plt.show()

        return f"Plotted {y} vs {x} as a {plot_type}."

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns.
        Returns the correlation coefficient and p-value.
        """

        error = self._ensure_loaded()

        if error:
            return error

        if col1 not in self.df.columns:
            return {"error": f"Column '{col1}' was not found."}

        if col2 not in self.df.columns:
            return {"error": f"Column '{col2}' was not found."}

        pearson_r, p_value = pearsonr(
            self.df[col1],
            self.df[col2]
        )

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(pearson_r), 4),
            "p_value": round(float(p_value), 4)
        }


print("Class defined")


# ============================================================
# Q4: Test CsvManager and Register CSV Tools
# ============================================================

# Initialize the CSV manager.
RESOURCES_DIR = Path("resources")

csv_backend = CsvManager(RESOURCES_DIR)


# Q4 test: Load bike_commute.csv directly.
print("\n--- Q4: Test CsvManager ---")

load_result = csv_backend.load_csv("bike_commute.csv")

print("Load result:", load_result)


# Compute the correlation between traffic density and average speed.
correlation_result = csv_backend.compute_correlation(
    "avg_traffic_density",
    "avg_speed_kmh"
)

print("Correlation result:", correlation_result)


# Connect tool names to their Python methods.
node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation,
}


# JSON schemas describing the tools available to the agent.
tools_schema = [

    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },

    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },

    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {
                        "type": "string",
                        "description": "Column name for y-axis."
                    },
                    "x": {
                        "type": "string",
                        "description": "Optional column name for x-axis."
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation and p-value between two columns in the active CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "Name of the first column.",
                    },
                    "col2": {
                        "type": "string",
                        "description": "Name of the second column.",
                    },
                },
                "required": ["col1", "col2"],
            },
        },
    },

]


# ============================================================
# Q5: Define the Agent Response Cycle
# ============================================================

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run one ReAct agent cycle using a simple tool-using agent.

    REASON:
      The model decides whether it needs a tool.

    ACT:
      Run the requested Python tool.

    OBSERVE:
      Send the tool result back to the model.
    """

    messages.append({
        "role": "user",
        "content": user_text
    })

    def observe_tool_result(tool_call_id, result):

        content = (
            json.dumps(result, default=str)
            if not isinstance(result, str)
            else result
        )

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

        return tool_message

    for loop_idx in range(max_tool_rounds):

        # REASON: Ask the model what to do next.
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        # Add the assistant response to conversation history.
        assistant_entry = {
            "role": "assistant",
            "content": msg.content,
        }

        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                tc.model_dump() for tc in msg.tool_calls
            ]

        messages.append(assistant_entry)

        # No tool calls means the model has finished answering.
        if not msg.tool_calls:
            return msg.content

        # ACT and OBSERVE: Execute each requested tool.
        for tool_call in msg.tool_calls:

            name = tool_call.function.name

            tool_args = json.loads(
                tool_call.function.arguments or "{}"
            )

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)

            if fn is None:
                result = {
                    "error": f"Tool '{name}' not found."
                }

            else:
                try:
                    result = (
                        fn(**tool_args)
                        if tool_args
                        else fn()
                    )

                except Exception as e:
                    print(
                        f"Tool error in {name}: "
                        f"{type(e).__name__}: {e}"
                    )

                    result = {
                        "error": (
                            f"Tool '{name}' failed: "
                            f"{type(e).__name__}: {e}"
                        )
                    }

            # Add the tool result to conversation history.
            messages.append(
                observe_tool_result(tool_call.id, result)
            )

    return "I hit the tool-round limit. Try a simpler request."


# ============================================================
# Q5: Set Up the System Prompt and Query the Agent
# ============================================================

SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]

print("\n--- Q5: Correlation Analysis ---")

result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh."
)

print(result)


# ============================================================
# Q6: Print the Complete Conversation Messages
# ============================================================

print("\n--- Q6: Complete Conversation Messages ---")

print(json.dumps(messages, indent=2, default=str))


# Message roles:
#
# system: Gives the agent its instructions and behavior.
# user: Contains the question asked by the user.
# assistant: Contains the model's response or tool-call request.
# tool: Contains the result returned by a Python tool.

# ============================================================
# Lesson 04: smolagents
# Q7: Wrap compute_correlation as a smolagents tool
# ============================================================

# --- Q7: Define smolagents tools ---

# Reuse the CsvManager instance created in Q4.
csv_manager = csv_backend


@tool
def list_csv_files() -> dict:
    """List the CSV files available in the resources directory.

    Returns:
        A dictionary containing the available CSV filenames.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from the resources directory.

    Args:
        filename: The name of the CSV file to load.

    Returns:
        A dictionary containing the loading status and column names.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Get the column names from the currently loaded CSV.

    Returns:
        A list of column names or an error dictionary.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Calculate summary statistics for selected CSV columns.

    Args:
        columns: The columns to summarize. If None, summarize all columns.

    Returns:
        A dictionary containing summary statistics.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Calculate descriptive statistics for one CSV column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dictionary containing descriptive statistics.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(
    y: str,
    x: str | None = None,
    plot_type: str = "line",
) -> str | dict:
    """Create a plot using columns from the currently loaded CSV.

    Args:
        y: Column to display on the y-axis.
        x: Column to display on the x-axis. If None, use the row index.
        plot_type: Type of plot, either line or scatter.

    Returns:
        A message describing the result or an error.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Calculate the Pearson correlation between two CSV columns.

    Args:
        col1: Name of the first column.
        col2: Name of the second column.

    Returns:
        A dictionary containing the correlation coefficient and p-value.
    """
    return csv_manager.compute_correlation(col1, col2)


print("\n--- Q7: smolagents Tool Description ---")
print(compute_correlation.description)


# Comparison:
# In Q4, I manually defined a JSON schema with the function name,
# description, and parameter information.
# smolagents generates the tool description and input schema from
# the decorated function, its type hints, and its docstring.
# The developer needs to provide a descriptive function name,
# accurate type hints, and a clear Google-style docstring
# explaining the purpose, arguments, and return value.

# All tools available to both smolagents agents.
TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]

# --- Q8: Create the shared model ---

model_to_use = "gpt-4o-mini"

model = OpenAIServerModel(
    api_key=api_key,
    model_id=model_to_use,
)

SYSTEM_PROMPT = (
    "You are a small data assistant that analyzes files stored in resources/. "
    "Use the available tools to complete requested tasks. "
    "Do not guess. Keep answers short and student-friendly."
)

# Create the tool-based agent.
tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
    instructions=SYSTEM_PROMPT,
)

# Instructions for the code-based agent.
CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can:
1. Call the provided tools.
2. Write and execute Python code when the available tools are not enough.

Rules:
- Prefer available tools for simple tasks.
- If the user requests plot styling such as colors, markers,
  titles, labels, or grid settings that plot_data cannot control,
  write matplotlib code directly instead.
- Do not claim that you completed something unless the code
  or tool actually completed it.
- Assume the active dataset is available through csv_manager.df
  after a CSV has been loaded.
"""

# Create the code-based agent.
code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=[
        "pandas",
        "matplotlib.pyplot",
        "numpy",
    ],
    max_steps=8,
)

# --- Q8: Compare ToolCallingAgent and CodeAgent ---

prompt = (
    "Load bike_commute.csv. Plot avg_heart_rate vs duration_min "
    "as a scatter plot with green dots."
)

print("\n--- Q8: ToolCallingAgent Response ---")
response_tool = tool_agent.run(prompt)
print(response_tool)

print("\n--- Q8: CodeAgent Response ---")
response_code = code_agent.run(
    prompt,
    additional_args={"csv_manager": csv_manager},
)
print(response_code)

## Q8 Reflection:
#
# Both agents successfully loaded bike_commute.csv and created
# a scatter plot of avg_heart_rate vs duration_min.
#
# The ToolCallingAgent used the predefined load_csv and
# plot_data tools. Its plot_data tool does not have a parameter
# for changing the dot color.
#
# The CodeAgent also used the predefined tools instead of
# generating custom matplotlib code. Its execution log did
# not show an explicit green color setting either.
#
# Although both agents reported that the plot was created
# with green dots, their execution logs do not demonstrate
# that the requested color was applied.
#
# This shows that ToolCallingAgent is useful for tasks that
# match predefined tools. CodeAgent can be more flexible
# because it can generate Python code, but it may still choose
# to use existing tools when they appear sufficient.
# --- Q9: Final Reflection ---
#
# 1. ToolCallingAgent is a better choice for a task such as
# loading a CSV file or calculating the correlation between
# two columns. These tasks are already supported by predefined
# tools. The agent can select and call the appropriate tool
# without needing to generate new code.
#
# 2. A meaningful risk of CodeAgent is that it generates and
# executes Python code. The generated code could contain errors
# or perform unintended operations. ToolCallingAgent is more
# limited because it can only use the tools provided by the
# developer.
