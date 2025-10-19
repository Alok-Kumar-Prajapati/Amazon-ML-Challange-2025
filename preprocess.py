import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup
import joblib

# -----------------------------
# 1. Load CSVs
# -----------------------------
train_df = pd.read_csv("../dataset/train.csv")
test_df = pd.read_csv("../dataset/test.csv")

# -----------------------------
# 2. Text Feature Parsing
# -----------------------------
def parse_catalog_text(text):
    title, bullets, description = "", "", ""
    
    # Title
    title_match = re.search(r"Item Name:\s*(.*?)(?:\n|$)", text, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
    
    # Bullet Point N:
    bullet_matches = re.findall(r"Bullet Point \d+:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    bullet_text = " ".join([b.strip() for b in bullet_matches])
    
    # Product Description
    desc_match = re.search(r"Product Description:\s*(.*?)(?:\nValue:|$)", text, re.IGNORECASE | re.DOTALL)
    html_desc = desc_match.group(1).strip() if desc_match else ""
    
    soup = BeautifulSoup(html_desc, "html.parser")
    
    # <li> bullets
    li_bullets = [li.get_text(separator=" ").strip() for li in soup.find_all("li")]
    li_text = " ".join(li_bullets)
    
    # Clean description
    description = soup.get_text(separator=" ").strip()
    
    # Combine Bullet Point N and <li> bullets (remove duplicates)
    all_bullets = " ".join(list(dict.fromkeys(bullet_text.split(". ") + li_text.split(". "))))
    
    return title, all_bullets, description

for df in [train_df, test_df]:
    df[['title', 'bullets', 'description']] = df['catalog_content'].apply(
        lambda x: pd.Series(parse_catalog_text(str(x)))
    )
    df['text_input'] = df['title'] + " [SEP] " + df['bullets'] + " [SEP] " + df['description']

# -----------------------------
# 3. Numeric Feature Extraction
# -----------------------------
def extract_numeric_features(text):
    value_match = re.search(r"Value:\s*([\d\.]+)", text)
    value = float(value_match.group(1)) if value_match else np.nan
    
    unit_match = re.search(r"Unit:\s*([\w\s]+)", text, re.IGNORECASE)
    unit_raw = unit_match.group(1).strip().lower() if unit_match else "unknown"
    
    unit_mapping = {
        "oz": "Ounce", "ounce": "Ounce", "fl oz": "Fl Oz", "fl": "Fl Oz",
        "g": "Gram", "kg": "Kilogram", "count": "Count"
    }
    unit = unit_mapping.get(unit_raw, unit_raw.title())
    
    pack_match = re.search(r"Pack of (\d+)|Set of (\d+)", text, re.IGNORECASE)
    pack_size = int(pack_match.group(1) or pack_match.group(2)) if pack_match else 1
    
    weight_in_g = np.nan
    if unit == "Ounce": weight_in_g = value * 28.3495
    elif unit == "Fl Oz": weight_in_g = value * 29.57
    elif unit == "Gram": weight_in_g = value
    elif unit == "Kilogram": weight_in_g = value * 1000
    
    text_lower = text.lower()
    is_food = int(bool(re.search(r"\b(food|snack|cookies|soup|sauce|chutney|milk|tea|cheese|seasoning|dressing|beverages)\b", text_lower)))
    is_drink = int(bool(re.search(r"\b(drink|beverage|wine|tea|juice|milk)\b", text_lower)))
    is_cosmetic = int(bool(re.search(r"\b(soap|cream|powder|shampoo|lotion)\b", text_lower)))
    
    return pd.Series([value, unit, pack_size, weight_in_g, is_food, is_drink, is_cosmetic])

for df in [train_df, test_df]:
    df[['value', 'unit', 'pack_size', 'weight_in_g', 'is_food', 'is_drink', 'is_cosmetic']] = \
        df['catalog_content'].apply(lambda x: extract_numeric_features(str(x)))

# -----------------------------
# 4. Text Length / Bullet Features
# -----------------------------
for df in [train_df, test_df]:
    df['title_length'] = df['title'].apply(lambda x: len(x.split()))
    df['num_bullets'] = df['bullets'].apply(lambda x: len(x.split('. ')))
    df['avg_bullet_len'] = df['bullets'].apply(lambda x: np.mean([len(b.split()) for b in x.split('. ')]) if x else 0)

# -----------------------------
# 5. Missing Value Handling
# -----------------------------
numeric_cols = ['value', 'pack_size', 'weight_in_g', 'title_length', 'num_bullets', 'avg_bullet_len']
for df in [train_df, test_df]:
    for col in numeric_cols:
        df[f'{col}_missing'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0)

# -----------------------------
# 6. Standardize Numeric Features
# -----------------------------
scaler = StandardScaler()
train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])  # Use train mean/std

# Save scaler for future use
joblib.dump(scaler, "../numeric_scaler.pkl")

# -----------------------------
# 7. Final Feature Set
# -----------------------------
feature_cols = ['text_input', 'unit', 'pack_size', 'weight_in_g', 
                'is_food', 'is_drink', 'is_cosmetic', 'title_length', 'num_bullets', 'avg_bullet_len']

final_train = train_df[feature_cols + ['sample_id', 'price']]
final_test = test_df[feature_cols + ['sample_id']]

final_train.to_parquet("../processed_train.parquet", index=False)
final_test.to_parquet("../processed_test.parquet", index=False)

print("✅ Preprocessing complete.")
print("Train sample:")
print(final_train.head())
print("Test sample:")
print(final_test.head())
