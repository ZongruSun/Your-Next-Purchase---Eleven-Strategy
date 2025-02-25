import os
os.system("pip install streamlit")
os.system("pip install torch")

import streamlit as st
import torch
import pickle
from model import LSTMRecommender, recommend, load_data

# Configuration de l'interface
st.set_page_config(page_title="Recommandation de Produits", layout="wide")

# Chargement des données et du modèle
st.sidebar.title("🔄 Chargement des données...")

# Charger les données
df1, df_stocks, category_family_encoder = load_data()

# Charger l'encodeur de catégories
with open("category_encoder.pkl", "rb") as f:
    category_family_encoder = pickle.load(f)

num_categories = len(category_family_encoder.classes_)

# Initialiser le modèle et le charger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRecommender(num_categories).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Interface utilisateur avec Streamlit
st.title("🛍️ Recommandation de Produits avec LSTM")
st.write("Entrez un ID client pour obtenir des recommandations de produits basées sur ses achats.")

# Entrée utilisateur pour l'ID du client
client_id = st.text_input("🔍 **Entrez l'ID du client**", value="39370740138294")

if st.button("📌 Obtenir des recommandations"):
    if client_id.isdigit():
        client_id = int(client_id)
        top_categories, correct_count, actual_categories, total_available, recommended_products = recommend(
            client_id, model, df1, df_stocks, category_family_encoder, k=3
        )

        # Affichage des résultats
        st.subheader("📊 Résultats de la recommandation")
        st.write(f"**Client ID :** {client_id}")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### 🔝 Catégories recommandées")
            for cat in top_categories:
                st.write(f"✅ {cat}")

            st.write(f"🎯 **Catégories bien prédites** : {correct_count} / {len(actual_categories)}")
            st.write(f"📌 **Catégories réellement achetées** : {', '.join(actual_categories)}")

        with col2:
            st.write("### 🏷️ Produits recommandés")
            if len(recommended_products) > 0:
                for prod in recommended_products:
                    st.write(f"🛒 Produit ID : {prod}")
            else:
                st.write("⚠️ Aucun produit disponible en stock pour ce client.")

            st.write(f"📦 **Produits disponibles après filtrage** : {total_available}")

    else:
        st.error("❌ Veuillez entrer un ID client valide (nombre entier).")

# Exécution : `streamlit run app.py`
