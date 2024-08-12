# data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

class DataPreprocessor:
    @staticmethod
    def preprocess(df):
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df = df.copy()

        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(r'[^a-zA-Z0-9]', '_', regex=True)

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Infer and convert data types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric, if fails, leave as object
                df[col] = pd.to_numeric(df[col], errors='ignore')
                
                # If still object, try to convert to datetime
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass

        # Identify column types
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        date_columns = df.select_dtypes(include=['datetime64']).columns

        # Handle missing values
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].mean())

        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')

        # Handle outliers (capping at 3 standard deviations)
        for col in numeric_columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = df[col].clip(mean - 3*std, mean + 3*std)

        # Encode categorical variables
        for col in categorical_columns:
            df = pd.get_dummies(df, columns=[col], prefix=col, dummy_na=True)

        # Scale numeric features
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        # Feature selection (remove low variance features)
        selector = VarianceThreshold()
        df = pd.DataFrame(selector.fit_transform(df), columns=df.columns[selector.get_support()])

        return df

    @staticmethod
    def prepare_for_embedding(df):
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df = df.copy()
        # Convert all columns to string
        for col in df.columns:
            df[col] = df[col].astype(str)
        # Combine all columns into a single string for each row
        return df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)