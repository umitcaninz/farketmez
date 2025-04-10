import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pymcdm.methods import EDAS, CODAS, WASPAS
from pyDecision.algorithm import idocriw_method
import numpy as np
import traceback

# Kafe verilerini işleyecek ana sınıf
class CafeAnalyzer:
    def __init__(self):
        self.df = None
        self.df_sayisal_veri = None
        self.ranks_df = None
        self.merged_df = None

    def process_data(self, data):
        try:
            print("Processing cafe data...")
            if not data:
                print("Error: Empty data received")
                return None

            # Convert data to DataFrame directly
            processed_data = []
            for cafe in data:
                try:
                    processed_cafe = {
                        'kafe_adi': str(cafe.get('name', '')),
                        'genel_puan': float(cafe.get('rating', 0)),
                        'yiyecek_puani': float(cafe.get('detailedRatings', {}).get('food', 0)),
                        'hizmet_puani': float(cafe.get('detailedRatings', {}).get('service', 0)),
                        'atmosfer_puani': float(cafe.get('detailedRatings', {}).get('atmosphere', 0)),
                        'mesafe': round(float(cafe.get('distance', 0)), 2),
                        'toplam_yorum': int(cafe.get('totalReviews', 0))
                    }
                    processed_data.append(processed_cafe)
                except Exception as e:
                    print(f"Error processing cafe: {str(e)}")
                    continue

            if not processed_data:
                print("Error: No valid data to process")
                return None

            # Create DataFrame directly from processed data
            self.df = pd.DataFrame(processed_data)
            print(f"Created DataFrame with {len(self.df)} rows")

            # Set cafe name as index
            self.df.set_index('kafe_adi', inplace=True)

            # Extract numerical data
            self.df_sayisal_veri = self.df[["genel_puan", "yiyecek_puani", "hizmet_puani", "atmosfer_puani", "mesafe", "toplam_yorum"]]

            # Normalize data using MinMaxScaler
            scaler = MinMaxScaler()
            self.df_sayisal_veri[["genel_puan", "yiyecek_puani", "hizmet_puani", "atmosfer_puani", "mesafe", "toplam_yorum"]] = \
                scaler.fit_transform(self.df_sayisal_veri[["genel_puan", "yiyecek_puani", "hizmet_puani", "atmosfer_puani", "mesafe", "toplam_yorum"]])
            
            df_sayisal_veri_values = self.df_sayisal_veri.values

            # Define criterion types (max/min)
            criterion_type = ['max', 'max', "max", "max", "min", "max"]
            criterion_type_2 = [1, 1, 1, 1, -1, 1]

            # Process data to handle zero values
            islenmis_veri = self.veri_hazirla(df_sayisal_veri_values)
            
            # Calculate weights using IDOCRIW method
            agirliklar = idocriw_method(islenmis_veri, criterion_type, verbose=False)

            # Apply MCDM methods
            methods = [WASPAS(), CODAS()]
            method_names = ['WASPAS ', 'CODAS']
            prefs = []
            ranks = []

            for method in methods:
                pref = method(islenmis_veri, agirliklar, criterion_type_2)
                rank = method.rank(pref)
                prefs.append(pref)
                ranks.append(rank)

            # Create alternatives and preference DataFrames
            alternatives = np.arange(0, len(df_sayisal_veri_values))
            prefs_df = pd.DataFrame(zip(*prefs), columns=method_names, index=alternatives)
            self.ranks_df = pd.DataFrame(zip(*ranks), columns=method_names, index=alternatives)

            # Calculate Copeland scores
            copeland_scores = self.calculate_copeland_scores(alternatives)
            prefs_df["CopelandScore"] = pd.Series(copeland_scores)
            self.ranks_df["COPELAND"] = prefs_df["CopelandScore"].rank(ascending=False).astype(int)

            # Final processing
            self.ranks_df.index = self.df.index
            self.ranks_df.sort_values(by="COPELAND", ascending=True, inplace=True)
            
            # Merge results
            common_index = self.ranks_df.index.intersection(self.df.index)
            self.merged_df = pd.concat([self.ranks_df.loc[common_index], self.df.loc[common_index]], axis=1)
            self.merged_df.drop(columns=["WASPAS ", "CODAS"], inplace=True)
            self.merged_df.rename(columns={"COPELAND": "Sıralama"}, inplace=True)

            print("\n=== İşlenmiş Kafe Verileri ===")
            print("\nSütunlar:")
            print(self.merged_df.columns.tolist())
            print("\nVeriler:")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_rows', None)
            print(self.merged_df)


            print("Data processing completed successfully")
            return self.merged_df.to_dict('index')

        except Exception as e:
            print(f"Error in process_data: {str(e)}")
            traceback.print_exc()
            return None

    @staticmethod
    def veri_hazirla(veri_dizisi):
        epsilon = np.finfo(float).eps
        return np.where(veri_dizisi == 0, epsilon, veri_dizisi)

    def calculate_copeland_scores(self, alternatives):
        copeland_scores = {alt: 0 for alt in alternatives}
        for alt1 in alternatives:
            for alt2 in alternatives:
                if alt1 != alt2:
                    wins = sum(self.ranks_df.loc[alt1] < self.ranks_df.loc[alt2])
                    losses = sum(self.ranks_df.loc[alt1] > self.ranks_df.loc[alt2])
                    copeland_scores[alt1] += wins - losses
        return copeland_scores 