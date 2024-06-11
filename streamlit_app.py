import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://raw.githubusercontent.com/sdrcr74/test/main/bank.csv'
bank = pd.read_csv(url)
st.title("Bank Marketing Campaign")
st.sidebar.title("Sommaire")
pages=["Contexte du projet","Exploration des données","Analyse des données","Modélisation","Conclusion"]
page=st.sidebar.radio("Aller à la page:", pages)
if page==pages[0]:
  st.write("Contexte du projet")
  st.write("L’objectif du projet est d’établir un modèle permettant de prédire le succès d’une campagne marketing d’une banque. Concrètement, il s’agit de prédire si suite à la campagne, un client souscrit ou non au produit Dépôt à terme.")
  st.write("Le jeu de données qui nous a été mis à disposition s’appelle “Bank Marketing Dataset”. Ce jeu de données est disponible librement sur Kaggle, mais à la base il vient de la UC Irvine Machine Learning Repository. Ce sont des données liées aux campagnes de marketing direct d’une banque portugaise. Il date de 2012.Il contient 11 162 lignes de données et 17 colonnes.")
  st.write("Dans un premier temps, nous étudierons les différentes variables puis analyserons le dataset et procéderons à un nettoyage des données: doublons, données manquantes, pertinence des différentes variables.")
  st.image("banking.jpg")
elif page==pages[1]:
  st.write('Exploration des données')
  st.write("Avant d'explorer les données du dataset, il nous a semblé pertinent de comprendre les différentes variables présentes dans le jeu de données.") 
  st.write("Pour la plupart, l'intitulé des variables était clair et compréhensible. Nous allons cependant clarifier certaines variables:") 
  st.write("-balance: montant du compte en banque")
  st.write("-housing: prêt immobilier") 
  st.write("-loan: autre prêt") 
  st.write("-contact: moyen de contact") 
  st.write("-day & month: jour et mois du contact")
  st.write("-duration: durée du contact") 
  st.write("-campaign: nombre de contact durant la campagne marketing") 
  st.write("-pdays: nombre de jours de contact avant la campagne")
  st.write("-previous: nombre de contact avant la campagne")
  st.write("-poutcome: résultat de la dernière campagne")
  st.write("Aperçu de notre dataset")
  st.dataframe(bank.head())
  st.write('Dimensions du Dataframe')
  st.write(bank.shape)
  if st.checkbox("Afficher le nombre de doublons"):
    st.dataframe(bank.duplicated())
  if st.checkbox("Afficher les valeurs manquantes"):
    st.dataframe(bank.isna().sum())
  if st.checkbox("Répartition de la variable deposit"):
    st.dataframe(bank['deposit'].value_counts())
  if st.checkbox("Répartition en % par résultat de la dernière campagne via la variable poutcome"):
    st.dataframe(bank['poutcome'].value_counts()/len(bank))
  if st.checkbox("Pourcentage du nombre de contact lors de la dernière campagne égal à 0"):
    st.dataframe((bank['previous'] == 0).value_counts()/len(bank))
  if st.checkbox("Pourcentage de -1 dans la variable pdays"):
    st.dataframe((len(bank[bank['pdays'] == -1]) / len(bank)))
  if st.checkbox("Nombre de chiffres négatifs dans la variable balance"):
    st.dataframe(len(bank[bank['balance'] < 0]))
elif page==pages[2]:
  st.write("Analyse des données")
  Graphique_sélectionné=st.sidebar.selectbox(label="Graphique", options=['Répartition par âge','Répartition par métier','Répartition par statut marital','Répartition par éducation','Répartition par mois','Répartition par défauts de paiement', 'Répartition par prêt immobilier','Répartition des prêts à la conso','Répartition par type de contact','Résultat sur la dernière campagne marketing','Répartition du nombre de dépôts à terme','Répartition du nombre de contact de la dernière campagne'])
  if Graphique_sélectionné =='Répartition par âge':   
    fig=sns.displot(x='age', data=bank)
    plt.title('Répartition par âge')
    st.pyplot(fig)
  if Graphique_sélectionné =='Répartition par métier':    
    fig1=sns.displot(x='job', data=bank)
    plt.xticks(rotation=90)
    plt.title('Répartition par métier')
    st.pyplot(fig1)
  if Graphique_sélectionné =='Répartition par statut marital': 
    fig2=sns.displot(x='marital', data=bank)
    plt.title('Répartition par statut marital')
    st.pyplot(fig2)
  if Graphique_sélectionné =='Répartition par éducation': 
    fig3=sns.displot(x='education', data=bank)
    plt.title('Répartition par éducation')
    st.pyplot(fig3)
  if Graphique_sélectionné =='Répartition par mois': 
    fig4=sns.displot(x='month', data=bank)
    plt.title('Répartition par mois')
    st.pyplot(fig4)
  if Graphique_sélectionné =='Répartition par défauts de paiement': 
    fig5=sns.displot(x='default', data=bank)
    plt.title('Répartition par défauts de paiement')
    st.pyplot(fig5)
  if Graphique_sélectionné =='Répartition par prêt immobilier': 
    fig6=sns.displot(x='housing',data=bank)
    plt.title('Répartition par prêt immobilier')
    st.pyplot(fig6)
  if Graphique_sélectionné =='Répartition des prêts à la conso': 
    fig7=sns.displot(x='loan', data=bank)  
    plt.title('Répartition des prêts à la conso')
    st.pyplot(fig7)
  if Graphique_sélectionné =='Répartition par type de contact': 
    fig8=sns.displot(x='contact', data=bank, stat = 'percent')
    plt.title('Répartition par type de contact')
    st.pyplot(fig8)
  if Graphique_sélectionné =='Résultat sur la dernière campagne marketing': 
    fig9=sns.displot(x='poutcome', data=bank, stat = 'percent')
    plt.title('Résultat sur la dernière campagne marketing')
    st.pyplot(fig9)
  if Graphique_sélectionné =='Répartition du nombre de dépôts à terme': 
    fig10=sns.displot(x='deposit', data=bank, stat = 'percent')
    plt.title('Répartition du nombre de dépôts à terme')
    st.pyplot(fig10)
  if Graphique_sélectionné =='Répartition du nombre de contact de la dernière campagne': 
    fig11=sns.displot(x='previous', data=bank, stat = 'percent')
    plt.title('Répartition du nombre de contact de la dernière campagne')
    st.pyplot(fig11)
  fig12=px.scatter(bank,x="balance",y="age", color='deposit', title='Relation Age, balance et Deposit')
  st.plotly_chart(fig12)


elif page==pages[3]:
  st.write("Modélisation")
  modèle_sélectionné=st.selectbox(label="Modèle", options=['Régression logistique','Decision Tree','Random Forest'])

elif page==pages[4]:
  st.write("Conclusion")
