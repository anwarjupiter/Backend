import pandas as pd
from langchain_ibm import ChatWatsonx
from langchain_ibm import WatsonxEmbeddings
from langchain_experimental.agents import create_pandas_dataframe_agent
from constants import *

# 1. Load Dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Temperature (\u00b0C)'] = pd.to_numeric(df['Temperature (\u00b0C)'], errors='coerce')
    return df

# 2. Detect Anomalies
def detect_anomalies(df):
    # Energy outliers
    q1 = df['Energy Consumption (kWh)'].quantile(0.25)
    q3 = df['Energy Consumption (kWh)'].quantile(0.75)
    iqr = q3 - q1
    energy_outliers = df[(df['Energy Consumption (kWh)'] < q1 - 1.5 * iqr) |
                         (df['Energy Consumption (kWh)'] > q3 + 1.5 * iqr)]

    # Temperature outliers
    temp_df = df.dropna(subset=['Temperature (\u00b0C)'])
    tq1 = temp_df['Temperature (\u00b0C)'].quantile(0.25)
    tq3 = temp_df['Temperature (\u00b0C)'].quantile(0.75)
    tiqr = tq3 - tq1
    temp_outliers = temp_df[(temp_df['Temperature (\u00b0C)'] < tq1 - 1.5 * tiqr) |
                            (temp_df['Temperature (\u00b0C)'] > tq3 + 1.5 * tiqr)]

    return energy_outliers, temp_outliers

# 3. Build Prompt for Analysis
def build_prompt(df, energy_outliers, temp_outliers):
    fan_count = len(df[(df['Equipment'] == 'Fan') & (df['Floor'] == 'F1')])
    total_rows = len(df)
    equipment_types = df['Equipment'].unique().tolist()

    prompt = f"""
You are a helpful assistant analyzing building equipment data. 
Here is the dataset summary:

- Total entries: {total_rows}
- Equipment types: {equipment_types}
- Number of fans on the first floor: {fan_count}
- Energy consumption outliers: {len(energy_outliers)}
- Temperature outliers: {len(temp_outliers)}

Energy outlier examples:
{energy_outliers[['Building', 'Floor', 'Space', 'Equipment', 'Energy Consumption (kWh)']].head(3).to_string(index=False)}

Temperature outlier examples:
{temp_outliers[['Building', 'Floor', 'Space', 'Equipment', 'Temperature (°C)']].head(3).to_string(index=False)}

Please provide a detailed summary, identifying any potential equipment issues or anomalies worth investigating.
"""
    return prompt

# 4. Run Analysis with LLM
def analyze_with_llm(prompt):
    llm = ChatWatsonx(
        model_id=MODEL_GRANITE_8B,
        project_id=WATSONX_PROJECT_ID,
        apikey=WATSONX_API_KEY,
        url=SERVER_URL,
        params=WASTSONX_PARAMS
    )
    response = llm.invoke(prompt)
    return response

# 5. Full Pipeline
def run_analysis(file_path):
    df = load_data(file_path)
    energy_outliers, temp_outliers = detect_anomalies(df)
    prompt = build_prompt(df, energy_outliers, temp_outliers)
    analysis = analyze_with_llm(prompt)
    return analysis

# Example Usage
if __name__ == "__main__":
    file_path = "input/civil.csv"  # Path to your CSV
    result = run_analysis(file_path)
    print("\n--- Analysis Result ---\n")
    print(result)
