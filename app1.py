import numpy as np
import pickle
import streamlit as st
import sklearn
import xgboost

# Load the model and scaler (ensure these files are generated from regno1.ipynb)
with open('modelreg.pkl', 'rb') as model_reg:
    model = pickle.load(model_reg)

# Modify the prediction function for regression
def prediction(uranium_lead_ratio, carbon_14_ratio, radioactive_decay_series, stratigraphic_layer_depth,
               geological_period, paleomagnetic_data, inclusion_of_other_fossils, isotopic_composition,
               surrounding_rock_type, stratigraphic_position, fossil_size, fossil_weight):
    
    # Updated categorical feature mappings
    geological_period_mapping = {
        'Cambrian': 0, 'Triassic': 10, 'Cretaceous': 2, 'Devonian': 3, 'Jurassic': 4,
        'Paleogene': 7, 'Permian': 8, 'Neogene': 5, 'Ordovician': 6, 'Carboniferous': 1, 'Silurian': 9
    }
    geological_period = geological_period_mapping[geological_period]

    paleomagnetic_data_mapping = {'Normal polarity': 0, 'Reversed polarity': 1}
    paleomagnetic_data = paleomagnetic_data_mapping[paleomagnetic_data]

    surrounding_rock_mapping = {
        'Sandstone': 2, 'Limestone': 1, 'Shale': 3, 'Conglomerate': 0
    }
    surrounding_rock_type = surrounding_rock_mapping[surrounding_rock_type]

    stratigraphic_position_mapping = {'Bottom': 0, 'Middle': 1, 'Top': 2}
    stratigraphic_position = stratigraphic_position_mapping[stratigraphic_position]

    # Boolean mapping
    inclusion_of_other_fossils = 1 if inclusion_of_other_fossils else 0

    # Combine the inputs into a feature array
    features = np.array([[uranium_lead_ratio, carbon_14_ratio, radioactive_decay_series, stratigraphic_layer_depth,
                          geological_period, paleomagnetic_data, inclusion_of_other_fossils, isotopic_composition,
                          surrounding_rock_type, stratigraphic_position, fossil_size, fossil_weight]])

    # Make the prediction
    predicted_age = model.predict(features)

    return predicted_age[0]  # Return the continuous predicted age

# Streamlit interface for the regression model
def main():
    st.set_page_config(page_title="Fossil Age Prediction", page_icon="🦴", layout="centered")

    st.markdown("<div class='main-title'>Fossil Age Prediction App</div>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        # Collect inputs for all the features
        uranium_lead_ratio = st.slider("Uranium-Lead Ratio", min_value=0.0, max_value=1.4, step=0.01)
        carbon_14_ratio = st.slider("Carbon-14 Ratio", min_value=0.0, max_value=0.9, step=0.01)
        radioactive_decay_series = st.slider("Radioactive Decay Series", min_value=0.0, max_value=1.6, step=0.01)
        stratigraphic_layer_depth = st.slider("Stratigraphic Layer Depth (m)", min_value=0.0, max_value=494.2, step=0.01)
        fossil_size = st.slider("Fossil Size (cm)", min_value=0.13, max_value=216.39, step=0.1)
        fossil_weight = st.slider("Fossil Weight (kg)", min_value=0.62, max_value=1010.09, step=0.1)
        isotopic_composition = st.slider("Isotopic Composition", min_value=0.0, max_value=3.08, step=0.01)

        # Updated categorical inputs
        geological_period = st.selectbox('Geological Period', (
            'Cambrian', 'Triassic', 'Cretaceous', 'Devonian', 'Jurassic',
            'Paleogene', 'Permian', 'Neogene', 'Ordovician', 'Carboniferous', 'Silurian'
        ))
        paleomagnetic_data = st.selectbox('Paleomagnetic Data', ('Normal polarity', 'Reversed polarity'))
        inclusion_of_other_fossils = st.checkbox('Inclusion of Other Fossils')
        surrounding_rock_type = st.selectbox('Surrounding Rock Type', ('Sandstone', 'Limestone', 'Shale', 'Conglomerate'))
        stratigraphic_position = st.selectbox('Stratigraphic Position', ('Bottom', 'Middle', 'Top'))

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Call the prediction function
        predicted_age = prediction(uranium_lead_ratio, carbon_14_ratio, radioactive_decay_series, stratigraphic_layer_depth,
                                   geological_period, paleomagnetic_data, inclusion_of_other_fossils, isotopic_composition,
                                   surrounding_rock_type, stratigraphic_position, fossil_size, fossil_weight)

        # Display the predicted fossil age
        st.markdown(f"<div class='result'>The predicted fossil age is {predicted_age:.2f} million years</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()