import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import requests
from io import BytesIO
import zipfile
import io


DATA_PATH = 'data'


def merge_latlong(df):
    latlong_df = pd.read_csv(f'{DATA_PATH}/uscities.csv')
    latlong_df = latlong_df[['City', 'lat', 'lng']]
    merged_df = pd.merge(df, latlong_df, on='City', how='inner')
    return merged_df.drop_duplicates(subset=['City'])


def get_targetability_df(crime_weight, wealth_weight, minimum_population):
    crime_df = pd.read_excel(f'{DATA_PATH}/Table_8_Offenses_Known_to_Law_Enforcement_by_State_by_City_2019.xls', skiprows=3).iloc[:-8]
    crime_df['State'] = crime_df['State'].str.replace('\d+', '', regex=True)
    crime_df['City'] = crime_df['City'].str.replace('\d+', '', regex=True)
    crime_df['City'] = crime_df['City'].str.replace(',', '')
    crime_df['State'] = crime_df['State'].ffill()

    crime_df.columns = crime_df.columns.str.replace('\n', ' ', regex=False).str.replace('\d+', '', regex=True)
    crime_df = crime_df[['State', 'City', 'Population', 'Robbery', 'Property crime', 'Burglary', 'Arson']]
    crime_df['State'] = crime_df['State'].str.title()
    # crime_df['Total residential crime'] = crime_df[['Robbery', 'Property crime', 'Burglary', 'Arson']].sum(axis=1)
    crime_df['Total residential crime'] = crime_df[['Property crime']].sum(axis=1)

    crime_df['Residential crime per capita'] = crime_df['Total residential crime'] / crime_df['Population']

    population_threshold = minimum_population
    crime_df = crime_df[crime_df['Population'] >= population_threshold]

    crime_df = crime_df.sort_values(by='Residential crime per capita', ascending=False).reset_index(drop=True)


    income_df = pd.read_csv(f'{DATA_PATH}/ACSST1Y2021.S1901-Data.csv', skiprows=1)

    columns_to_keep = [
        'Geographic Area Name',
        'Estimate!!Households!!Total',
        # 'Estimate!!Households!!Total!!$100,000 to $149,999',
        # 'Estimate!!Households!!Total!!$150,000 to $199,999',
        'Estimate!!Households!!Total!!$200,000 or more',
    ]
    income_df = income_df[columns_to_keep]

    columns_to_check = columns_to_keep[2:]
    income_df = income_df.drop(income_df[income_df[columns_to_check].eq('N').any(axis=1)].index)

    for col in income_df.columns:
        if col != 'Geographic Area Name':
            income_df[col] = income_df[col].astype(float)

    income_df['State'] = income_df['Geographic Area Name'].str.split(',').str[1].str.strip()
    income_df.insert(0, 'State', income_df.pop('State'))
    income_df['Geographic Area Name'] = income_df['Geographic Area Name'].str.replace(r',.*$', '').str.rstrip(' city')
    income_df = income_df.rename(columns={'Geographic Area Name': 'City'})

    income_df['Total target households'] = income_df[columns_to_check].sum(axis=1)
    income_df.drop(columns=columns_to_check, inplace=True)

    income_df = income_df.sort_values(by='Total target households', ascending=False).reset_index(drop=True)
    income_df['City'] = income_df['City'].str.split(',').str[0].str.replace(' city', '')





    df = pd.merge(crime_df, income_df, on=['State', 'City'])
    df = df[['State', 'City', 'Population', 'Total residential crime', 'Residential crime per capita', 'Estimate!!Households!!Total', 'Total target households']]

    columns_to_normalize = ['Total target households', 'Residential crime per capita']
    scaler = MinMaxScaler()
    df[[f'Normalized {col}' for col in columns_to_normalize]] = scaler.fit_transform(df[columns_to_normalize])

    df['Target viability'] = (df['Normalized Total target households']**wealth_weight) * (df['Normalized Residential crime per capita']**crime_weight)

    df.insert(2, 'Target viability', df.pop('Target viability'))
    df['Targetable households (by wealth)'] = df['Estimate!!Households!!Total'] * df['Total target households']
    df = df[['State', 'City', 'Population', 'Target viability', 'Targetable households (by wealth)', 'Total residential crime']]

    scaler = MinMaxScaler(feature_range=(0, 100))
    target_viability_values = df['Target viability'].values.reshape(-1, 1)
    normalized_target_viability = scaler.fit_transform(target_viability_values)
    df['Target viability'] = normalized_target_viability
    df = df.rename(columns={'Target viability': 'Target viability'})

    df = df.sort_values(by='Target viability', ascending=False).reset_index(drop=True)
    df.index = range(1, len(df) + 1)

    return df