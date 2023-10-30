import pickle
import streamlit as st

model = pickle.load(open('estimasi_Round_Played.sav', 'rb'))

st.title('Estimasi Round Played')

Kill = st.number_input('Masukan Total Kill', min_value=1, step=1)
Death = st.number_input('Masukan Total Death Played', min_value=1, step=1)
Rounds_Win = st.number_input('Masukan Total Kemenangan Team', min_value=1, step=1)
Rounds_Lose = st.number_input('Masukan Total Kekalahan Team', min_value=1, step=1)
KD = st.number_input('Masukan Total Kill Death Played')
HS = st.number_input('Masukan Ratio Headshot')

predict = ''

if st.button('Estimasi Round Played'):
    predict = model.predict(
        [[ Kill, Death, Rounds_Win, Rounds_Lose, KD, HS]]
    )
    st.write ('Estimasi Round Played : ', predict)