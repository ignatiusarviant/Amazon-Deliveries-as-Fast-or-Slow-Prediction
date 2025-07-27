import pandas as pd
from geopy.distance import geodesic

dataset = pd.read_csv("C:\\Users\\HP\\Downloads\\Python Only\\Delivery Time ML\\amazon_delivery.csv")
df = pd.DataFrame(dataset)
del df["Order_ID"]
df = df[df["Vehicle"] == "van"]
df = df[df["Category"] == "Electronics"]
df = df[df["Area"] == "Metropolitian "]
df[["Store_Latitude","Store_Longitude","Drop_Latitude","Drop_Longitude"]] = df[["Store_Latitude","Store_Longitude", 
                                                                            "Drop_Latitude", "Drop_Longitude"]].replace(0, pd.NA)
df = df.dropna().reset_index(drop=True)
df = df.reset_index(drop=True)


def calculate_distance(row):
    store = (row["Store_Latitude"], row["Store_Longitude"])
    drop = (row["Drop_Latitude"], row["Drop_Longitude"])
    return round(geodesic(store, drop).km, 2)

def preprocess():
    weather_class = {'Sunny': 0, 'Stormy': 1, 'Sandstorms': 2, 'Cloudy': 3, 'Fog': 4, 'Windy': 5}
    traffic_class = {'High ':0, 'Jam ': 1, 'Low ': 2, 'Medium ': 3}
    van_class = {'van': 0}
    metropolitan_class = {'Metropolitian ': 0}
    category_class = {'Electronics': 0}

    df["Area"] = df["Area"].map(metropolitan_class)
    df["Category"] = df["Category"].map(category_class)
    df["Vehicle"] = df["Vehicle"].map(van_class)
    df["Weather"] = df["Weather"].map(weather_class)
    df["Traffic"] = df["Traffic"].map(traffic_class)
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    return df

preprocess()
df["Distance_km"] = df.apply(calculate_distance, axis=1)

def is_fast_or_not():
    median = df["Delivery_Time"].median()
    df["fast_or_not"] = df["Delivery_Time"].apply(lambda x: 0 if x < median else 1)
    return df["fast_or_not"]

is_fast_or_not()
df.to_csv("Data Training & Test.csv", index=False)
