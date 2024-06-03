import pickle
import streamlit as st
import numpy as np

# Membaca model dan scaler
mushroom_model = pickle.load(open('mushroom_classifier.sav', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Mapping kategori ke keterangan
cap_shape_mapping = {'x': 'Convex', 'f': 'Flat', 's': 'Sunken', 'b': 'Bell', 'o': 'Others', 'p': 'Spherical', 'c': 'Conical'}
cap_surface_mapping = {'t': 'Sticky', 's': 'Smooth', 'y': 'Scaly', 'h': 'Shiny', 'g': 'Grooves', 'd': 'Dotted', 'e': 'Fleshy', 'k': 'Silky', 'i': 'Fibrous', 'w': 'Wrinkled', 'l': 'Leathery'}
cap_color_mapping = {'n': 'Brown', 'y': 'Yellow', 'w': 'White', 'g': 'Gray', 'e': 'Red', 'o': 'Orange', 'r': 'Green', 'u': 'Purple', 'p': 'Pink', 'b': 'Buff', 'k': 'Black', 'l': 'Blue'}
stem_color_mapping = {'w': 'White', 'n': 'Brown', 'y': 'Yellow', 'g': 'Grey', 'o': 'Orange', 'e': 'Red', 'u': 'Purple', 'f': 'None', 'p': 'Pink', 'k': 'Black', 'r': 'Green', 'l': 'Blue', 'b': 'Buff'}
has_ring_mapping = {'f': 'False', 't': 'True'}
habitat_mapping = {'d': 'Woods', 'g': 'Grasses', 'l': 'Leaves', 'm': 'Meadows', 'h': 'Heaths', 'w': 'Waste', 'p': 'Paths', 'u': 'Urban'}
season_mapping = {'a': 'Autumn', 'u': 'Summer', 'w': 'Spring', 's': 'Winter'}

# Judul web
st.title('Mushroom Prediction')

# Mengambil input dari user
col1, col2 = st.columns(2)
with col1:
    Cap_Diameter = st.text_input('Cap Diameter (cm)')
    Stem_Height = st.text_input('Stem Height (cm)')
    Stem_Width = st.text_input('Stem Width (mm)')
    Cap_Shape = st.selectbox('Cap Shape', list(cap_shape_mapping.values()))
    Cap_Surface = st.selectbox('Cap Surface', list(cap_surface_mapping.values()))
with col2:
    Cap_Color = st.selectbox('Cap Color', list(cap_color_mapping.values()))
    Stem_Color = st.selectbox('Stem Color', list(stem_color_mapping.values()))
    Has_Ring = st.selectbox('Has Ring', list(has_ring_mapping.values()))
    Habitat = st.selectbox('Habitat', list(habitat_mapping.values()))
    Season = st.selectbox('Season', list(season_mapping.values()))

# Code untuk prediksi 
mushroom_class = ''

# Membuat tombol prediksi
if st.button('Prediction Result'):
    try:
        # Konversi input yang relevan menjadi float
        Cap_Diameter = float(Cap_Diameter)
        Stem_Height = float(Stem_Height)
        Stem_Width = float(Stem_Width)
        
        # One-hot encoding untuk input kategori
        Cap_Shape_encoded = [0] * 7
        Cap_Surface_encoded = [0] * 11
        Cap_Color_encoded = [0] * 12
        Stem_Color_encoded = [0] * 13
        Has_Ring_encoded = [0] * 2
        Habitat_encoded = [0] * 8
        Season_encoded = [0] * 4

        # Mapping input kategori ke index yang sesuai
        Cap_Shape_encoded[list(cap_shape_mapping.values()).index(Cap_Shape)] = 1
        Cap_Surface_encoded[list(cap_surface_mapping.values()).index(Cap_Surface)] = 1
        Cap_Color_encoded[list(cap_color_mapping.values()).index(Cap_Color)] = 1
        Stem_Color_encoded[list(stem_color_mapping.values()).index(Stem_Color)] = 1
        Has_Ring_encoded[list(has_ring_mapping.values()).index(Has_Ring)] = 1
        Habitat_encoded[list(habitat_mapping.values()).index(Habitat)] = 1
        Season_encoded[list(season_mapping.values()).index(Season)] = 1
        
        # Menggabungkan semua input menjadi satu array
        input_data = [Cap_Diameter, Stem_Height, Stem_Width] + Cap_Shape_encoded + Cap_Surface_encoded + Cap_Color_encoded + Stem_Color_encoded + Has_Ring_encoded + Habitat_encoded + Season_encoded

        # Hanya skalakan input numerik
        numeric_features = scaler.transform([[Cap_Diameter, Stem_Height, Stem_Width]])
        
        # Gabungkan fitur numerik yang telah diskalakan dengan fitur kategorikal yang telah di-one-hot encoding
        input_data_scaled = list(numeric_features[0]) + Cap_Shape_encoded + Cap_Surface_encoded + Cap_Color_encoded + Stem_Color_encoded + Has_Ring_encoded + Habitat_encoded + Season_encoded
        
        # Lakukan prediksi
        mushroom_prediction = mushroom_model.predict([input_data_scaled])

        # Tentukan kelas jamur berdasarkan prediksi
        if np.array_equal(mushroom_prediction[0], [0, 1]):
            mushroom_class = 'Poisonous Mushroom'
        else:
            mushroom_class = 'Edible Mushroom'

        st.success(mushroom_class)
    except ValueError as e:
        st.error(f'Error dalam konversi input: {e}')
