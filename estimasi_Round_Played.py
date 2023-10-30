import pickle
import streamlit as st

model = pickle.load(open('estimasi_Round_Played.sav', 'rb'))

st.title('Estimasi Round Played')

Kill = st.number_input('Masukan Total Kill')
Death = st.number_input('Masukan Total Death Played')
Rounds_Win = st.number_input('Masukan Total Kemenangan Team')
Rounds_Lose = st.number_input('Masukan Total Kekalahan Team')
KD = st.number_input('Masukan Total Kill Death Played')
HS = st.number_input('Masukan Ratio Headshot')

predict = ''

if st.button('Estimasi Round Played'):
    predict = model.predict(
        [[ 'Kill', 'Death', 'Rounds Win', 'Rounds Lose', 'K/D', 'HS %']]
    )
    st.write ('Estimasi Round Played : ', predict)