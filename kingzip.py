import pandas as pd

data = pd.read_csv("Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")
king_county_data = data[data["CountyName"] == "King county"]
print(king_county_data)
