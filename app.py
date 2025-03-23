import streamlit as st
import requests
import json

st.title("💵 Prédiction de Crédit")
st.markdown("Remplis les informations ci-dessous pour tester le modèle.")

# Champs visibles par l'utilisateur (en positif)
AMT_INCOME_TOTAL = st.number_input("Revenu total", min_value=0.0, value=10000.0)
AMT_CREDIT = st.number_input("Montant du crédit", min_value=0.0, value=200000.0)
AMT_ANNUITY = st.number_input("Montant de l'annuité", min_value=0.0, value=70000.0)
CNT_FAM_MEMBERS = st.number_input("Nombre de membres dans la famille", min_value=1.0, step=1.0, value=6.0)

# Champs en jours, mais affichés en valeur positive
AGE_DAYS = st.number_input("Âge (en jours)", value=9000.0, min_value=0.0)
EMPLOYED_DAYS = st.number_input("Jours d'emploi", value=0.0, min_value=0.0)
REGISTRATION_DAYS = st.number_input("Jours depuis inscription", value=100.0, min_value=0.0)
ID_PUBLISH_DAYS = st.number_input("Jours depuis publication ID", value=300.0, min_value=0.0)

if st.button("🔍 Prédire"):
    payload = {
        "features": [{
            "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
            "AMT_CREDIT": AMT_CREDIT,
            "AMT_ANNUITY": AMT_ANNUITY,
            "CNT_FAM_MEMBERS": CNT_FAM_MEMBERS,
            # Conversion en négatif
            "DAYS_BIRTH": -AGE_DAYS,
            "DAYS_EMPLOYED": -EMPLOYED_DAYS,
            "DAYS_REGISTRATION": -REGISTRATION_DAYS,
            "DAYS_ID_PUBLISH": -ID_PUBLISH_DAYS
        }]
    }

    response = requests.post(
        "https://impl-mentez-un-modele-de-scoring.onrender.com/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        try:
            result = response.json()
            decision = result['decision'][0]
            probability = result["probability"][0]

            if decision == "Crédit accordé":
                st.success(f"✅ Résultat : {decision}")
                st.markdown(
                    f"<span style='color:green'>Probabilité : {probability:.2%}</span>",
                    unsafe_allow_html=True
                )
            else:
                st.error(f"❌ Résultat : {decision}")
                st.markdown(
                    f"<span style='color:red'>Probabilité : {probability:.2%}</span>",
                    unsafe_allow_html=True
                )
        except json.JSONDecodeError:
            st.error("❌ Erreur : réponse non JSON.")
            st.write(response.text)

