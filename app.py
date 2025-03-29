import streamlit as st
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Prédiction de Crédit", layout="wide")

# Chargement des données clients
@st.cache_data
def load_clients():
    return pd.read_csv("clients.csv")  # Ce fichier doit être présent avec les colonnes utiles

clients_df = load_clients()

# Démarrage de l'application

st.title("💼 Dashboard de Prédiction de Crédit")

# Tabs : Vue client / Comparaison / Simulation
tab1, tab2, tab3 = st.tabs(["🔍 Client", "📊 Comparaison", "✍️ Simulation"])

########################
# Onglet 1 : Visualisation client
########################
with tab1:
    st.header("🔍 Analyse d’un client")
    client_id = st.selectbox("Choisir un client", options=clients_df["SK_ID_CURR"].unique())

    client_data = clients_df[clients_df["SK_ID_CURR"] == client_id].to_dict(orient="records")[0]
    st.subheader("🧾 Informations client")
    st.json(client_data)

    # Construction du payload API
    features = {k: client_data[k] for k in [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "CNT_FAM_MEMBERS", "DAYS_BIRTH", "DAYS_EMPLOYED",
        "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"
    ]}
    payload = {"features": [features]}

    with st.spinner("⏳ Prédiction en cours..."):
        try:
            response = requests.post(
                "https://impl-mentez-un-modele-de-scoring.onrender.com/predict",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            result = response.json()

            st.success(f"Résultat : {result['decision'][0]}")
            st.metric("Probabilité d'accord", f"{result['probability'][0]*100:.2f}%")

            if "shap_values" in result:
                st.subheader("📈 Interprétation du score (SHAP)")
                shap_data = result["shap_values"][0]
                shap_series = pd.Series(shap_data).sort_values()
                fig, ax = plt.subplots()
                shap_series.plot(kind="barh", ax=ax)
                ax.set_title("Impact des variables sur la décision")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur API : {e}")

########################
# Onglet 2 : Comparaison
########################
with tab2:
    st.header("📊 Comparaison avec d’autres clients")
    variable = st.selectbox("Choisir une variable à comparer", options=[
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "CNT_FAM_MEMBERS"
    ])

    fig, ax = plt.subplots()
    ax.hist(clients_df[variable], bins=30, alpha=0.7, label="Tous les clients")
    ax.axvline(client_data[variable], color='red', linestyle='dashed', linewidth=2, label="Client sélectionné")
    ax.set_title(f"Distribution de {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Nombre de clients")
    ax.legend()
    st.pyplot(fig)
    st.caption("Graphique accessible : les couleurs contrastées et les légendes aident à la lecture.")

########################
# Onglet 3 : Simulation (optionnel)
########################
with tab3:
    st.header("✍️ Simulation / Nouveau dossier")
    income = st.number_input("Revenu total", value=client_data["AMT_INCOME_TOTAL"])
    credit = st.number_input("Montant du crédit", value=client_data["AMT_CREDIT"])
    annuity = st.number_input("Annuité", value=client_data["AMT_ANNUITY"])
    members = st.number_input("Membres de la famille", value=client_data["CNT_FAM_MEMBERS"])
    age = st.number_input("Age (en jours)", value=-client_data["DAYS_BIRTH"])
    employed = st.number_input("Jours d'emploi", value=-client_data["DAYS_EMPLOYED"])
    registration = st.number_input("Jours depuis inscription", value=-client_data["DAYS_REGISTRATION"])
    id_pub = st.number_input("Jours depuis ID", value=-client_data["DAYS_ID_PUBLISH"])

    if st.button("🔄 Rafraîchir la prédiction"):
        new_payload = {
            "features": [{
                "AMT_INCOME_TOTAL": income,
                "AMT_CREDIT": credit,
                "AMT_ANNUITY": annuity,
                "CNT_FAM_MEMBERS": members,
                "DAYS_BIRTH": -age,
                "DAYS_EMPLOYED": -employed,
                "DAYS_REGISTRATION": -registration,
                "DAYS_ID_PUBLISH": -id_pub
            }]
        }
        with st.spinner("Calcul de la nouvelle prédiction..."):
            try:
                response = requests.post(
                    "https://impl-mentez-un-modele-de-scoring.onrender.com/predict",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(new_payload)
                )
                result = response.json()
                st.success(f"Résultat : {result['decision'][0]}")
                st.metric("Probabilité", f"{result['probability'][0]*100:.2f}%")
            except Exception as e:
                st.error(f"Erreur API : {e}")

