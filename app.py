import streamlit as st
import requests
import json

st.title("💵 Prédiction de Crédit")

st.markdown("Remplis les informations ci-dessous pour tester le modèle.")

# Formulaire utilisateur
AMT_INCOME_TOTAL = st.number_input("Revenu total", min_value=0.0)
AMT_CREDIT = st.number_input("Montant du crédit", min_value=0.0)
AMT_ANNUITY = st.number_input("Montant de l'annuité", min_value=0.0)
CNT_FAM_MEMBERS = st.number_input("Nombre de membres dans la famille", min_value=1.0, step=1.0)
DAYS_BIRTH = st.number_input("Âge en jours (ex: -12000)", value=-12000.0)
DAYS_EMPLOYED = st.number_input("Jours d'emploi (ex: -3000)", value=-3000.0)
DAYS_REGISTRATION = st.number_input("Jours depuis l'inscription", value=-4000.0)
DAYS_ID_PUBLISH = st.number_input("Jours depuis publication ID", value=-3500.0)

# Envoyer la requête
if st.button("🔍 Prédire"):
    payload = {
        "features": [{
            "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
            "AMT_CREDIT": AMT_CREDIT,
            "AMT_ANNUITY": AMT_ANNUITY,
            "CNT_FAM_MEMBERS": CNT_FAM_MEMBERS,
            "DAYS_BIRTH": DAYS_BIRTH,
            "DAYS_EMPLOYED": DAYS_EMPLOYED,
            "DAYS_REGISTRATION": DAYS_REGISTRATION,
            "DAYS_ID_PUBLISH": DAYS_ID_PUBLISH
        }]
    }

    response = requests.post(
        "https://impl-mentez-un-modele-de-scoring.onrender.com/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        result = response.json()
        st.success(f"✅ Résultat : {result['decision'][0]}")
        st.write("Probabilité :", result["probability"][0])
    else:
        st.error(f"❌ Erreur : {response.json()['error']}")
