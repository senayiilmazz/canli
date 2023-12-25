import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.preprocessing import RobustScaler
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    
st.markdown(
    """
    <style>
        body {
            background-color: #8ea3d8; 
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Suicide Prediction 🖤 ")

#st.set_page_config(background_color=(142, 163, 216))



# Spotify API bilgileri
client_id = "821e87bf233045be9e933d4f7c247ac8"
client_secret = "a06ed36fd2074d4291b602be96e51a2b"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

log_model = joblib.load('logistic_regression_model.joblib')


def data_prep_for_prediction(dataframe):
    # Sütun düşürme
    columns_to_drop = [ 'uri', 'type', 'track_href', 'analysis_url']
    dataframe.drop(columns_to_drop, axis=1, inplace=True)

    # Aykırı değerleri düzeltme
    replace_with_thresholds(dataframe, "speechiness")
    replace_with_thresholds(dataframe, "liveness")
    replace_with_thresholds(dataframe, "duration_ms")
    replace_with_thresholds(dataframe, "time_signature")

    # Özellikleri RobustScaler kullanarak ölçekleme
    scaler = RobustScaler()
    columns_to_scale = ['tempo', 'duration_ms', 'time_signature', 'loudness', 'key']
    dataframe[columns_to_scale] = scaler.fit_transform(dataframe[columns_to_scale])

    # Bağımsız değişkenleri ayarlama
    X = dataframe.drop([ "id"], axis=1)

    return X

# Şarkı tahminleme işlemi
def predict_suicide(song_name, artist_name, gender, model):
    # Spotify'dan şarkı arama
    result = sp.search(q=f"track:{song_name} artist:{artist_name}", type='track', limit=1)

    if result['tracks']['items']:
        track = result['tracks']['items'][0]

        # Şarkının audio features'larını alma ve hazırlama
        audio_features = sp.audio_features(tracks=[track['id']])[0]
        audio_features['gender'] = gender
        X_song = pd.DataFrame([audio_features])  # Veriyi DataFrame'e çevirme
        X_song = data_prep_for_prediction(X_song)  # Şarkı verisini hazırlama

        # Modelle tahmin yapma (suicide tahmini)
        suicide_prediction = model.predict(X_song)

        # Şarkının kapak resmini gösterme (sayfanın kenarında)
        st.sidebar.image(track['album']['images'][0]['url'], caption=f'Şarkı Kapak Resmi: {song_name}',
                         use_column_width=True)

        # Tahmin değeri ve audio feature'ları göster
        st.success(f'Tahmin (Suicide): {suicide_prediction}')
        st.text('Kullanılan Audio Features:')
        st.write(audio_features)

        return suicide_prediction[0]
    else:
        st.error('Şarkı bulunamadı.')
        return None

def main():

    # Kullanıcıdan şarkı adını al
    song_name = st.text_input('Şarkı Adı:')

    # Kullanıcıdan şarkıcı adını al
    artist_name = st.text_input('Şarkıcı Adı:')

    # Kullanıcıdan cinsiyeti al
    gender = st.radio('Cinsiyet Seçin:', ['Erkek', 'Kadın'])

    # Cinsiyeti sayısal bir değere çevirme
    gender_encoding = 0 if gender == 'Erkek' else 1

    if st.button('Tahminle'):
        # Şarkı tahminleme işlemi
        predict_suicide_result = predict_suicide(song_name, artist_name, gender_encoding, log_model)

        if predict_suicide_result is not None:
            st.success('Tahmin başarıyla yapıldı.')
        else:
            st.error('Şarkı bulunamadı.')

if __name__ == '__main__':
    main()
