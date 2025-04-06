import pandas as pd

# Chemin vers le fichier brut
input_file = "application_train.csv"

# Colonnes attendues par l'API
required_columns = [
    "SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "CNT_FAM_MEMBERS", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"
]

# Chargement et nettoyage
df = pd.read_csv(input_file)
df_clean = df[required_columns].dropna()

# On convertit les colonnes dans le bon format
df_clean = df_clean.astype({
    "SK_ID_CURR": int,
    "AMT_INCOME_TOTAL": float,
    "AMT_CREDIT": float,
    "AMT_ANNUITY": float,
    "CNT_FAM_MEMBERS": float,
    "DAYS_BIRTH": float,
    "DAYS_EMPLOYED": float,
    "DAYS_REGISTRATION": float,
    "DAYS_ID_PUBLISH": float,
})

# Échantillon de 1000 clients (modifiable)
df_sample = df_clean.sample(n=1000, random_state=42)

# Sauvegarde
output_file = "clients.csv"
df_sample.to_csv(output_file, index=False)

print(f"✅ Fichier '{output_file}' généré avec succès.")
