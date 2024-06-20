import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://raw.githubusercontent.com/sdrcr74/bank_nov23/main/bank.csv'
bank = pd.read_csv(url)
st.markdown("<img src='https://raw.githubusercontent.com/sdrcr74/bank_test/main/datascientest_logo.png' width='100' style='display: block; margin: 0 auto;'>" , unsafe_allow_html=True)
st.markdown("<h1 style='text-align : center;'>Prédiction du succès d une campagne de Marketing d’une banque</h1>", unsafe_allow_html=True)
st.subheader("NOV23_CONTINU_DA - Datascientest", divider='blue')

st.sidebar.title("Sommaire")
pages=["Généralités du projet","Le jeu de données", "Datavisualisation","Préparation des données","Modélisation","Conclusion"]
page=st.sidebar.radio("Aller à la page:", pages)
st.sidebar.title("Auteurs")
with st.sidebar:
        st.write("Maxence MALHERRE")
        st.write("Sophie DORCIER")
        st.write("Stéphane LASCAUX")
        st.write("Van-Anh HA")
if page==pages[0]:
  st.header("Généralités")
  st.markdown("- En marketing, l'analyse prédictive permet une personnalisation plus précise et des campagnes mieux ciblés.")
  st.markdown("- Application en projet de formation Data Analyst : prédiction du succès de la campagne marketing du produit Dépôt à terme d’une banque.")
  st.markdown("- Ce Streamlit illustre notre approche, allant de l'exploration des données à la création du modèle prédictif.")
elif page==pages[1]:
  st.subheader("Le jeu de données")
  st.markdown("- Le jeu de données nous a été fourni, il s'agit d'un fichier CSV s'appelant 'Bank Marketing Dataset', disponible librement sur Kaggle.")
  st.markdown("- Description du contenu : Données personnelles (informations démographiques, situation financière, contact précédent avec la banque) issues de campagnes d'appel télémarketing d’une banque portugaise.")
  st.markdown("- Périmètre temporel : 2012")
  st.markdown("- Source : UC Irvine Machine Learning Repository")
  st.markdown("- Dimension : 11 162 lignes & 17 colonnes (16 variables explicatives & 1 variable cible")
  st.markdown("- Définition des variables :")
  url2 = 'https://raw.githubusercontent.com/vha22/bank-marketing/main/Liste%20variable.csv'
  liste_variable = pd.read_csv(url2)
  st.dataframe(liste_variable)
  st.markdown("- Qualité de données : à première vue, la base de données nous semble propre :")
  if st.checkbox("Nombre de doublons :"):
    st.write(bank.duplicated().sum())
  if st.checkbox("Nombre de valeurs manquantes :"):
    st.write(bank.isna().any(axis = 0))
  st.write("Dans la partie suivante, nous allons explorer les données de façon plus approfondie à travers de la datavisualisation.")

elif page==pages[5]:
  st.subheader("Conclusion")
  st.markdown("- Ce rapport offre à une entreprise des perspectives précieuses sur les schémas comportementaux des clients et les déterminants influençant leurs choix, offrant ainsi la possibilité d'optimiser les stratégies de marketing et de prendre des décisions stratégiques mieux éclairées.")
  st.markdown("- Notre travail a permis d'identifier les variables les plus significatives tout en surmontant des obstacles tels que la signification des variables.")
  st.markdown("- Des améliorations supplémentaires auraient pu être envisagées afin de peaufiner davantage notre modèle (analyse plus approfondie de l'importance des variables, application de techniques avancées de réduction de la dimensionnalité)")      
