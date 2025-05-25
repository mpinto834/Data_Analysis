import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Goal: Evaluate how weather conditions affect accident frequency and severity in 2021

# --- File paths ---
dataset_path = 'dataset/'
accidents_path = dataset_path + 'US_Accidents_March23.csv'
weather_path = dataset_path + 'WeatherEvents_Jan2016-Dec2022.csv'

# --- Load full datasets (adjust if needed for performance) ---
print('📥 Loading datasets. ..')
accidents = pd.read_csv(accidents_path)
weather = pd.read_csv(weather_path)

# --- Convert date columns ---
weather['StartTime(UTC)'] = pd.to_datetime(weather['StartTime(UTC)'], errors='coerce')
accidents['Start_Time'] = pd.to_datetime(accidents['Start_Time'], errors='coerce')

# --- Filter only data from the year 2021 ---
weather_2021 = weather[weather['StartTime(UTC)'].dt.year == 2021].copy()
accidents_2021 = accidents[accidents['Start_Time'].dt.year == 2021].copy()

# --- Create rounded hour column to facilitate later merge ---
weather_2021['Hour'] = weather_2021['StartTime(UTC)'].dt.floor('H')
accidents_2021['Hour'] = accidents_2021['Start_Time'].dt.floor('H')

# --- Show basic information ---
print("2021 Weather Dataset Information:")
print(f"Number of records: {weather_2021.shape[0]:,}")
print("\nTypes of weather events in 2021:")
print(weather_2021['Type'].value_counts())
print("\nSeverity of events:")
print(weather_2021['Severity'].value_counts())

# --- Save filtered datasets for later use ---
weather_2021.to_csv('dataset/WeatherEvents_2021.csv', index=False)
accidents_2021.to_csv('dataset/US_Accidents_2021.csv', index=False)

print("\n2021 datasets successfully saved!")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# --- Initial diagnosis: missing values ---
print('\nPercentage of missing values per column (US Accidents):')
print((accidents_2021.isnull().mean() * 100).sort_values(ascending=False))

print('\nPercentage of missing values per column (Weather Events):')
print((weather_2021.isnull().mean() * 100).sort_values(ascending=False))

# --- Copies for processing ---
accidents_clean = accidents_2021.copy()
weather_clean = weather_2021.copy()

# --- Remove columns with >50% missing ---
def drop_high_missing(df, threshold=0.5):
    return df.loc[:, df.isnull().mean() < threshold]

accidents_clean = drop_high_missing(accidents_clean)
weather_clean = drop_high_missing(weather_clean)

# --- Missing value imputation ---
def simple_impute(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df

accidents_clean = simple_impute(accidents_clean)
weather_clean = simple_impute(weather_clean)

# --- Validation and conversion of 'Severity' column ---
if 'Severity' in accidents_clean.columns:
    print("\nUnique values in 'Severity' before filtering:", accidents_clean['Severity'].unique())
    
    accidents_clean = accidents_clean[accidents_clean['Severity'].isin([1, 2, 3, 4])]
    accidents_clean['Severity'] = accidents_clean['Severity'].astype(int)

    print("Unique values after cleaning:", accidents_clean['Severity'].unique())

# --- Post-cleaning check ---
print('\nAfter cleaning, missing percentage (US Accidents):')
print((accidents_clean.isnull().mean() * 100).sort_values(ascending=False))

print('\nAfter cleaning, missing percentage (Weather Events):')
print((weather_clean.isnull().mean() * 100).sort_values(ascending=False))

# --- Visualization: Severity distribution ---
if 'Severity' in accidents_clean.columns:
    plt.figure(figsize=(8, 4))
    unique_severities = sorted(accidents_clean['Severity'].unique())
    sns.countplot(x='Severity', data=accidents_clean, order=unique_severities)
    plt.title('Count by Severity (US Accidents)')
    plt.xlabel('Severity')
    plt.ylabel('Number of Accidents')
    plt.tight_layout()
    plt.savefig('plots/countplot_severity_us_accidents.png')
    plt.close()

# --- Save cleaned datasets ---
accidents_clean.to_csv('dataset/US_Accidents_clean.csv', index=False)
weather_clean.to_csv('dataset/WeatherEvents_clean.csv', index=False)

print('\nCleaned files saved as:')
print('→ dataset/US_Accidents_clean.csv')
print('→ dataset/WeatherEvents_clean.csv')

# --- Reload cleaned datasets ---
df_accidents = pd.read_csv('dataset/US_Accidents_clean.csv')
df_weather = pd.read_csv('dataset/WeatherEvents_clean.csv')

# --- Prepare and merge cleaned data ---
def prepare_and_merge_data(df_weather, df_accidents):
    df_weather = df_weather.rename(columns={
        'StartTime(UTC)': 'Weather_Start',
        'EndTime(UTC)': 'Weather_End',
        'ZipCode': 'Zip',
        'Severity': 'Weather_Severity'
    })
    df_accidents = df_accidents.rename(columns={
        'Start_Time': 'Accident_Start',
        'End_Time': 'Accident_End',
        'Zipcode': 'Zip',
        'Severity': 'Accident_Severity'
    })

    df_weather['Weather_Start'] = pd.to_datetime(df_weather['Weather_Start'], errors='coerce')
    df_accidents['Accident_Start'] = pd.to_datetime(df_accidents['Accident_Start'], errors='coerce')

    df_weather['Zip'] = df_weather['Zip'].astype(str).str[:5]
    df_accidents['Zip'] = df_accidents['Zip'].astype(str).str[:5]

    df_weather.dropna(subset=['Zip'], inplace=True)
    df_accidents.dropna(subset=['Zip'], inplace=True)

    df_weather_sorted = df_weather.sort_values(by='Weather_Start')
    df_accidents_sorted = df_accidents.sort_values(by='Accident_Start')

    merged = pd.merge_asof(
        df_accidents_sorted,
        df_weather_sorted,
        by='Zip',
        left_on='Accident_Start',
        right_on='Weather_Start',
        direction='backward',
        tolerance=pd.Timedelta('6h')
    )

    merged.dropna(subset=['EventId'], inplace=True)
    return merged

# Generate updated df_merged with the cleaned datasets
df_merged = prepare_and_merge_data(df_weather.copy(), df_accidents.copy())

df_merged.to_csv('dataset/df_merged.csv')

import os
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output folder exists
os.makedirs("plots", exist_ok=True)

# Final filter
columns_of_interest = [
    'Accident_Start', 'Zip', 'Type', 'Temperature(F)', 'Humidity(%)', 
    'Visibility(mi)', 'Accident_Severity', 'Weather_Severity'
]
df_analysis = df_merged[columns_of_interest].copy()

# Type conversions
df_analysis['Temperature(F)'] = pd.to_numeric(df_analysis['Temperature(F)'], errors='coerce')
df_analysis['Humidity(%)'] = pd.to_numeric(df_analysis['Humidity(%)'], errors='coerce')
df_analysis['Visibility(mi)'] = pd.to_numeric(df_analysis['Visibility(mi)'], errors='coerce')
df_analysis.dropna(inplace=True)

# Accident frequency by weather event type
plt.figure(figsize=(10, 4))
sns.countplot(y='Type', data=df_analysis, order=df_analysis['Type'].value_counts().index)
plt.title('Accident Frequency by Weather Event Type')
plt.xlabel('Number of Accidents')
plt.ylabel('Event Type')
plt.tight_layout()
plt.savefig('plots/accident_frequency_by_event_type.png')
plt.close()

# Average accident severity by weather event type
severity_by_weather = df_analysis.groupby('Type')['Accident_Severity'].mean().sort_values()
plt.figure(figsize=(10, 4))
severity_by_weather.plot(kind='barh')
plt.title('Average Accident Severity by Weather Event Type')
plt.xlabel('Average Severity')
plt.tight_layout()
plt.savefig('plots/average_severity_by_event_type.png')
plt.close()

# Boxplots: weather vs severity
climatic_vars = ['Visibility(mi)']
for var in climatic_vars:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Accident_Severity', y=var, data=df_analysis)
    plt.title(f'{var} vs Accident Severity')
    plt.xlabel('Accident Severity')
    plt.ylabel(var)
    plt.tight_layout()
    # Filename based on variable name
    filename = f"plots/boxplot_{var.lower().replace('(%)','pct').replace('(f)','f').replace('(mi)','mi')}_vs_severity.png"
    plt.savefig(filename)
    plt.close()

def clean_missing_and_outliers(df):
    required_cols = ['Temperature(F)', 'Weather_Condition', 'Precipitation(in)_x']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Remove rows with critical missing values
    df_clean = df.dropna(subset=required_cols)

    # --- Outlier Treatment with IQR (Temperature) ---
    temp = df_clean[['Temperature(F)', 'Humidity(%)', 'Visibility(mi)',
                                 'Precipitation(in)_x', 'Accident_Severity']]
    q1, q3 = temp.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    before = df_clean.shape[0]
    df_clean = df_clean[(temp >= lower) & (temp <= upper)]
    after = df_clean.shape[0]

    print(f"Removed {before - after} outliers based on Temperature(F) (IQR).")

    return df_clean

# Function call
df_cleaned = clean_missing_and_outliers(df_merged.copy())

# Summary description focusing on weather-related variables
summary = df_cleaned[['Temperature(F)', 'Precipitation(in)_x']].describe()
print("\nDescriptive Statistics (post-cleaning):")
print(summary)

# Frequency of weather conditions
print("\nTop 10 weather conditions:")
print(df_cleaned['Weather_Condition'].value_counts().head(10))

import os
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output folder exists
os.makedirs("plots", exist_ok=True)

# Visual style
plt.style.use('default')
sns.set_theme()

# Rename precipitation column for clarity
df_cleaned = df_cleaned.rename(columns={'Precipitation(in)_x': 'Precipitation(in)'})

# --- Distributions of climatic variables ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Temperature Histogram
sns.histplot(data=df_cleaned, x='Temperature(F)', ax=axes[0], bins=30, kde=True)
axes[0].set_title('Temperature Distribution')
axes[0].set_xlabel('Temperature (°F)')

# 2. Types of Weather Events
sns.countplot(data=df_cleaned, y='Type', order=df_cleaned['Type'].value_counts().index, ax=axes[1])
axes[1].set_title('Types of Weather Events')
axes[1].set_xlabel('Number of Occurrences')

# 3. Top 10 Weather Conditions
top_weather = df_cleaned['Weather_Condition'].value_counts().head(10).index
sns.countplot(data=df_cleaned[df_cleaned['Weather_Condition'].isin(top_weather)],
              y='Weather_Condition', ax=axes[2])
axes[2].set_title('Top 10 Weather Conditions')
axes[2].set_xlabel('Number of Occurrences')

plt.tight_layout()
plt.savefig("plots/complete_climatic_distributions.png")
plt.close()

# --- Accident severity by weather event type ---
type_counts = df_cleaned['Type'].value_counts()
valid_types = type_counts[type_counts >= 100].index
filtered = df_cleaned[df_cleaned['Type'].isin(valid_types)]

plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered, x='Type', y='Accident_Severity', order=valid_types)
plt.title('Accident Severity by Weather Event Type (N ≥ 100)')
plt.xticks(rotation=45)
plt.xlabel('Weather Event Type')
plt.ylabel('Accident Severity')
plt.tight_layout()
plt.savefig("plots/severity_by_event_type.png")
plt.close()


import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
os.makedirs("plots", exist_ok=True)

# --- Top 10 Weather Conditions with Most Accidents ---
plt.figure(figsize=(12, 6))
top_conditions = df_cleaned['Weather_Condition'].value_counts().head(10)
sns.barplot(x=top_conditions.values, y=top_conditions.index, palette='Blues_d')
plt.title("Top 10 Weather Conditions Associated with Accidents")
plt.xlabel("Number of Accidents")
plt.ylabel("Weather Condition")
plt.tight_layout()
plt.savefig("plots/top10_weather_conditions.png")
plt.close()

# --- Boxplot: Temperature by Weather Condition ---
plt.figure(figsize=(14, 6))
filtered = df_cleaned[df_cleaned['Weather_Condition'].isin(top_conditions.index)]
sns.boxplot(x='Weather_Condition', y='Temperature(F)', data=filtered, palette='coolwarm')
plt.title("Temperature Distribution by Weather Condition")
plt.xticks(rotation=45)
plt.xlabel("Weather Condition")
plt.ylabel("Temperature (°F)")
plt.tight_layout()
plt.savefig("plots/boxplot_temperature_by_condition.png")
plt.close()

# --- Boxplot: Precipitation by Weather Event Type ---
plt.figure(figsize=(10, 6))
top_types = df_cleaned['Type'].value_counts().head(5).index
sns.boxplot(data=df_cleaned[df_cleaned['Type'].isin(top_types)], 
            x='Type', 
            y='Precipitation(in)', palette='crest')
plt.title("Precipitation by Weather Event Type")
plt.xticks(rotation=45)
plt.xlabel("Weather Event Type")
plt.ylabel("Precipitation (in)")
plt.tight_layout()
plt.savefig("plots/boxplot_precipitation_by_event_type.png")
plt.close()

# --- Accident Severity by Weather Condition (Top 10) ---
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered, x='Weather_Condition', y='Accident_Severity', palette='flare')
plt.title("Accident Severity by Weather Condition (Top 10)")
plt.xticks(rotation=45)
plt.xlabel("Weather Condition")
plt.ylabel("Accident Severity")
plt.tight_layout()
plt.savefig("plots/severity_by_weather_condition_top10.png")
plt.close()