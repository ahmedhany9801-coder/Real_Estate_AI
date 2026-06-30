

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from groq import Groq
import requests
from streamlit_lottie import st_lottie
import time

##########################################################

model = joblib.load("best_model.pkl")

################################################################
st.markdown("""
    <style>

    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .block-container {
        padding-top: 0rem !important;
    }
    
    button[data-testid="baseButton-headerNoPadding"] {
        z-index: 9999 !important;
    }
    </style>
    """, unsafe_allow_html=True)


#############################################

st.set_page_config(layout='wide', page_title='RealEstate Ai')


######################################


st.markdown("""
<h1 style='color:white;text-align:center;'>
Know the real value of any home in seconds
</h1>
""", unsafe_allow_html=True)

###############

st.markdown("""
<style>
.stApp{
    background: #0B132B;
}
</style>
""", unsafe_allow_html=True)


#####################################
st.markdown("""
<style>

[data-testid="stVerticalBlock"]{
    background: transparent;
}

</style>
""", unsafe_allow_html=True)


####################################################################################


st.markdown("""
<style>

/* Sidebar Background */
[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#0B132B,#1C2541);
}

/* Navigation Title */
[data-testid="stSidebar"] label{
    color:white !important;
    font-weight:bold;
}

/* Radio Buttons */
div[role="radiogroup"] > label{
    background-color:#3A86FF;
    color:black !important;
    padding:12px;
    border-radius:15px;
    margin-bottom:10px;
    border:2px solid #E9C46A;
    font-weight:600;
}

/* Hover */###
div[role="radiogroup"] > label:hover{
    background-color:#4D96FF;
}

/* Links */
[data-testid="stSidebar"] a{
    color:#F4D35E !important;
    text-decoration:none;
}

/* Section Titles */
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5{
    color:white !important;
}

/* Text Area */
[data-testid="stSidebar"] textarea{
    background-color:#F8F9FA !important;
    color:black !important;
    border-radius:10px;
}

/* Placeholder */
[data-testid="stSidebar"] textarea::placeholder{
    color:gray !important;
}

/* Buttons */
.stButton > button{
    background-color:#3A86FF;
    color:white;
    border:none;
    border-radius:12px;
    font-weight:bold;
}

.stButton > button:hover{
    background-color:#4D96FF;
    color:white;
}###

/* Horizontal Lines */
hr{
    border:1px solid rgba(255,255,255,0.2);
}

</style>
""", unsafe_allow_html=True)

################
st.markdown("""
<style>

h1, h2, h3, h4, h5, h6 {
    color: white !important;
}


p {
    color: white !important;
}


.stSelectbox label,
.stSlider label,
.stNumberInput label,
.stTextInput label,
.stTextArea label,
.stRadio label,
.stToggle label {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

########################################################
st.image("REAL.png", width=700)
##############################
@st.cache_data
def load_data():
    return pd.read_csv("house_prices.csv")

df = load_data()
####################################################


with st.sidebar:

    st.image("Real_Estate_AI_2.png", use_container_width=True)

    st.markdown("""
    <h2 style='text-align:center;color:white;'>
    🏠 Real Estate AI
    </h2>

    <p style='text-align:center;color:#F4D35E;'>
    Smart Property Valuation Platform
    </p>
    """, unsafe_allow_html=True)
    
###################

    page = st.radio(
        "Navigation",
        ['Prediction', 'Data Insight📊', 'AI Asistant']
    )

#################
    st.markdown("---")

    st.markdown("### 📞To contact support")
    st.markdown("[📱 Call: 01284273509](tel:+201284273509)")
    st.markdown("[💬 WhatsApp](https://wa.me/201284273509)")
    st.markdown("[📧 Email](mailto:ahmedhany9801@gmail.com)")

    st.markdown("---")
    st.markdown("##### How was your experience today? ")
    rating = st.feedback("stars")

    if rating is not None:
        st.success(f" Thank you for your rating ({rating + 1} / 5) ⭐")
###############


    feedback_text = st.text_area("We value your feedback", placeholder=" Tell us what you think...")

    if st.button("Submit Feedback "):
        if feedback_text.strip() == "":
            st.warning(" Please provide some feedback before submitting .")
        else:
            st.success("Thank you! Your feedback has been received successfully. ✅")

#################################
if page == 'Prediction':


    st.subheader("Property Information🏡")

    col1, col2, col3 = st.columns(3)

#####################################################################################
    with col1:
        st.markdown("### Property Details")
        house_age = st.slider("House Age (Years)", 1, 79, 40)
        num_bedrooms = st.slider("Bedrooms", 1, 6, 3)
        num_bathrooms = st.slider("Bathrooms", 1, 5, 2)
        num_floors = st.slider("Floors", 1, 3, 1)
######################################################################################
    with col2:
        st.markdown("### Size & Outdoor")
        sqft_living = st.slider("Living Area (sqft)", 350, 6000, 1462, step=50)
        sqft_lot = st.slider("Lot Size (sqft)", 888, 12000, 5742, step=100)
        garage_type = st.selectbox("Garage Type", ['Single', 'Double'])
        has_pool = st.toggle("Swimming Pool")
########################################################################################
    with col3:
        st.markdown("### Location & Quality")
        neighborhood = st.selectbox(
            "Neighborhood",
            sorted(df['neighborhood'].dropna().unique())
        )
        house_condition = st.selectbox(
            "House Condition",
            ['Poor', 'Fair', 'Good', 'Excellent']
        )
        heating_type = st.selectbox(
            "Heating Type",
            sorted(df['heating_type'].dropna().unique())
        )
        school_rating = df[df['neighborhood']==neighborhood]['school_rating'].median()
        crime_rate = df[df['neighborhood']==neighborhood]['crime_rate'].median()


    fig_pool = px.pie(df, names='has_pool', hole=0.5, title='Swimming Pool Distribution')


    st.divider()
###################################################################################################
    predict_btn = st.button(" Predict House Price", use_container_width=True)

    if predict_btn:

        if sqft_living >= sqft_lot:
            st.error("⚠️ There is a problem : You must make (the Living Area (sqft)) smaller than (the Lot Size (sqft)).")

        else:
            total_rooms = num_bedrooms + num_bathrooms
            luxury_score = num_bathrooms / num_bedrooms if num_bedrooms > 0 else 0

            renovation_year = 0
            
            years_since_renovation = float(house_age)

            input_df = pd.DataFrame({
                'house_age': [house_age],
                'num_bedrooms': [num_bedrooms],
                'num_bathrooms': [num_bathrooms],
                'sqft_living': [sqft_living],
                'sqft_lot': [sqft_lot],
                'num_floors': [num_floors],
                'has_pool': [has_pool],
                'school_rating': [school_rating],
                'crime_rate': [crime_rate],
                'renovation_year': [renovation_year],
                'total_rooms': [total_rooms],
                'luxury_score': [luxury_score],
                'years_since_renovation': [years_since_renovation],
                'neighborhood': [neighborhood],
                'heating_type': [heating_type],
                'garage_type': [garage_type],
                'house_condition': [house_condition]
            })

            with st.spinner("🏠 Estimating property value..."):
                time.sleep(3)



                prediction = model.predict(input_df)[0]

                mae = 19541
                lower = prediction - mae
                upper = prediction + mae

                st.session_state['last_prediction'] = prediction
                st.session_state['lower'] = lower
                st.session_state['upper'] = upper
                st.session_state['user_inputs'] = input_df.to_dict('records')[0]

            st.success(f"🏠 Predicted Price: ${prediction:,.0f}")
            st.info(f"📊 Expected Range: ${lower:,.0f} - ${upper:,.0f}")

            st.markdown("---")

            st.success(
                " Need more insights📊? Use the ( AI Assistant 🔮) to analyze the predicted price and discover the key factors affecting the property's value."
            )
####################
    st.markdown("""
    <div style="
    text-align:center;
    padding:25px;
    border-radius:15px;
    background:rgba(255,255,255,0.05);
    ">

    <h4 style="color:white;">
    Connect With Me
    </h4>

    <p>
    💼 <a href="https://www.linkedin.com/in/ahmed-hany-722970415">LinkedIn</a>
    &nbsp;&nbsp;|&nbsp;&nbsp;

    🐙 <a href="https://github.com/ahmedhany9801-coder/Real_Estate_A">GitHub</a>
    </p>

    </div>
    """, unsafe_allow_html=True)



##########################################################################

elif page == 'Data Insight📊':

    
    st.markdown("""
    <h1 style='color:white'>
    Model Insights 📊
    </h1>
    """, unsafe_allow_html=True)
#######################################
    
    with st.expander(" Feature Importance ", expanded=False):
        try:
            preprocessor = model.named_steps['pre processing']
            xgb_model = model.named_steps['model']

            try:
                all_features = list(preprocessor.get_feature_names_out())
            except Exception:
                scaling_cols = ['house_age', 'num_bedrooms', 'num_bathrooms', 'sqft_living', 'num_floors', 'has_pool', 'total_rooms', 'luxury_score']
                median_cols = ['sqft_lot', 'crime_rate']
                mean_cols = ['school_rating']
                ord_col = ['garage_type', 'house_condition']
                nom_col = ['neighborhood', 'heating_type']

                ohe_features = (
                    preprocessor.named_transformers_['one hot']
                    .named_steps['on_hot']
                    .get_feature_names_out(nom_col)
                )
                all_features = scaling_cols + median_cols + mean_cols + ord_col + list(ohe_features)

            importances = xgb_model.feature_importances_

            min_length = min(len(all_features), len(importances))

            fi_df = pd.DataFrame({
                'Feature': all_features[:min_length],
                'Importance': importances[:min_length]
            })

            fi_df['Clean_Feature'] = fi_df['Feature'].apply(lambda x: x.split('__')[-1] if '__' in x else x)

            neighborhood_importance = fi_df[
                fi_df['Clean_Feature'].str.contains('neighborhood', case=False)
            ]['Importance'].sum()

            fi_df = fi_df[~fi_df['Clean_Feature'].str.contains('neighborhood', case=False)]

            fi_df = pd.concat([
                fi_df,
                pd.DataFrame({
                    'Feature': ['Neighborhood'],
                    'Clean_Feature': ['Neighborhood'],
                    'Importance': [neighborhood_importance]
                })
            ], ignore_index=True)

            heating_importance = fi_df[
                fi_df['Clean_Feature'].str.contains('heating_type', case=False)
            ]['Importance'].sum()

            fi_df = fi_df[~fi_df['Clean_Feature'].str.contains('heating_type', case=False)]

            fi_df = pd.concat([
                fi_df,
                pd.DataFrame({
                    'Feature': ['Heating Type'],
                    'Clean_Feature': ['Heating Type'],
                    'Importance': [heating_importance]
                })
            ], ignore_index=True)

            fi_df['Importance %'] = (fi_df['Importance'] / fi_df['Importance'].sum()) * 100
            fi_df = fi_df.sort_values('Importance %', ascending=False)

            feature_map = {
                'house_age': 'House Age',
                'num_bedrooms': 'Bedrooms',
                'num_bathrooms': 'Bathrooms',
                'sqft_living': 'Living Area',
                'num_floors': 'Floors',
                'has_pool': 'Has Pool',
                'total_rooms': 'Total Rooms',
                'luxury_score': 'Luxury Score',
                'sqft_lot': 'Lot Size',
                'crime_rate': 'Crime Rate',
                'school_rating': 'School Rating',
                'garage_type': 'Garage Type',
                'house_condition': 'House Condition',
                'Neighborhood': 'Neighborhood',
                'Heating Type': 'Heating Type'
            }
            fi_df['Display_Feature'] = fi_df['Clean_Feature'].map(feature_map).fillna(fi_df['Clean_Feature'])



#####
            df_plot = fi_df.head(10).sort_values('Importance %', ascending=True)

            fig = px.bar(
                df_plot,
                x='Importance %',
                y='Display_Feature',
                orientation='h',
                text='Importance %',
                title='Top Factors Affecting House Price',
                color='Display_Feature',
                color_discrete_sequence=px.colors.qualitative.Plotly
            )


            fig.update_layout(
                showlegend=False,
                yaxis={'categoryorder': 'array', 'categoryarray': df_plot['Display_Feature'].tolist()}
            )

            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )

            fig.update_layout(
                xaxis_title='Importance (%)',
                yaxis_title='',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as err:
            st.error(f"⚠️ Error computing feature importance: {err}")
    


########################################################################################################
    with st.expander(" Market Insights🏡"):

        st.markdown("##### Price Distribution")
        fig1 = px.histogram(df, x='price', nbins=50)
        st.plotly_chart(fig1, use_container_width=True)
#**************#

        st.markdown("##### Avg Price by Neighborhood")
        
        avg_neigh = df.groupby('neighborhood')['price'].mean().sort_values(ascending=False)
        colors = ['#1D9E75' if v == avg_neigh.max() else '#185FA5' for v in avg_neigh.values]
        fig2 = px.bar(avg_neigh, color=avg_neigh.index, color_discrete_sequence=colors)
        st.plotly_chart(fig2, use_container_width=True)


##############

        st.markdown("##### Avg Price by Condition")
        avg_cond = df.groupby('house_condition')['price'].mean().reindex(
            ['Poor', 'Fair', 'Good', 'Excellent']
        )
        fig3 = px.bar(avg_cond)
        st.plotly_chart(fig3, use_container_width=True)

##############################################################################

    with st.expander("ℹ️ Model Information"):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Model", "XGBoost")
        c2.metric("Test R²", "94.53%")
        c3.metric("Train R²", "96.23%")
        c4.metric("MAE", "$19,541")
        c5.metric("RMSE", "$24,826")
##########################################################################################################################

elif page == 'AI Asistant':

    st.subheader(" AI Assistant")

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    SYSTEM_PROMPT = """
You are an AI assistant inside the RealEstate AI application. Your job is to help the user understand the predicted price of their house, based on the machine learning model's output.

CONTEXT PROVIDED WITH EVERY QUESTION:
- PREDICTED PRICE, ERROR RANGE, MARKET AVERAGE
- PROPERTY DETAILS: neighborhood, house age, bedrooms, bathrooms, living area, lot size, floors, garage, swimming pool, house condition, heating type, school rating, crime rate, years since renovation

PRICE CLASSIFICATION (market average = $336,000):
- Cheap: below $252,000 (less than 75% of the average)
- Average: $252,000 to $420,000 (75% to 125% of the average)
- Expensive: above $420,000 (more than 125% of the average)


RESPONSE RULES:
1. When the user asks if the price is expensive or cheap, compare predicted_price to the ranges above and clearly state the classification.
2. When the user asks why the price is what it is, base your explanation strictly on the actual PROPERTY DETAILS values — not generic statements.
3. You MUST reference at least 2-3 specific property details (e.g. school rating, crime rate, neighborhood, size, number of floors) and explain how each one likely influenced the price, either pushing it up or down. Be specific and concrete, not generic or philosophical.
4. Only use the values present in PROPERTY DETAILS. Never invent or assume data that wasn't provided.
5. Mention the numeric difference from the market average when useful.
6. Keep your answer concise — one or two short paragraphs maximum.
7. If asked something unrelated to property valuation or the platform creator, politely redirect the conversation back to the app's purpose.
8. Do not provide legal or financial guarantees. Clarify that this is a machine learning estimate, not an official appraisal.

OUTPUT FORMAT (STRICT):
- Always respond in clear, natural English, regardless of the language the user wrote in.
- Write in well-spaced, properly formatted sentences — never merge words together.
- Never repeat the same sentence, phrase, or number more than once in your response.
- Do not restate the question or echo the context back to the user.
- Write the final answer once, directly, with no duplication or filler.
"""

    def build_context(predicted_price, lower, upper, user_inputs):
        neighborhood = user_inputs.get('neighborhood', 'Unknown')
        house_age = user_inputs.get('house_age', user_inputs.get('age', 'Unknown'))
        bedrooms = user_inputs.get('num_bedrooms', user_inputs.get('bedrooms', 'Unknown'))
        bathrooms = user_inputs.get('num_bathrooms', user_inputs.get('bathrooms', 'Unknown'))
        living_area = user_inputs.get('sqft_living', user_inputs.get('area', 'Unknown'))
        lot_size = user_inputs.get('sqft_lot', user_inputs.get('lot', 'Unknown'))
        floors = user_inputs.get('num_floors', user_inputs.get('floors', 'Unknown'))
        garage = user_inputs.get('garage_type', user_inputs.get('garage', 'Unknown'))

        pool_val = user_inputs.get('has_pool', 0)
        has_pool = 'Yes' if pool_val in [1, 'Yes', True] else 'No'

        condition = user_inputs.get('house_condition', user_inputs.get('condition', 'Good'))
        heating = user_inputs.get('heating_type', 'Standard')

        try:
            school_rating = f"{float(user_inputs.get('school_rating', 5.0)):.1f}"
        except Exception:
            school_rating = "Unknown"

        try:
            crime_rate = f"{float(user_inputs.get('crime_rate', 0.0)):.1f}"
        except Exception:
            crime_rate = "Unknown"

        years_since_renovation = user_inputs.get('years_since_renovation', 'Unknown')

        return f"""
PREDICTED PRICE: ${predicted_price:,.0f}
ERROR RANGE: ${lower:,.0f} - ${upper:,.0f}
MARKET AVERAGE: $336,000

PROPERTY DETAILS:
- Neighborhood: {neighborhood}
- House Age: {house_age} years
- Bedrooms: {bedrooms}
- Bathrooms: {bathrooms}
- Living Area: {living_area} sqft
- Lot Size: {lot_size} sqft
- Floors: {floors}
- Garage: {garage}
- Swimming Pool: {has_pool}
- House Condition: {condition}
- Heating Type: {heating}
- School Rating: {school_rating} / 10
- Crime Rate: {crime_rate} (lower is safer)
- Years Since Renovation: {years_since_renovation}

"""

    def ask_assistant(user_question, predicted_price, lower, upper, user_inputs):
        context = build_context(predicted_price, lower, upper, user_inputs)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\nUser question: {user_question}"}
            ],
            temperature=0.3,
            max_tokens=300
        )

        return response.choices[0].message.content

    if 'last_prediction' not in st.session_state:
        st.warning("You must make a prediction first from the Prediction page.")
    else:
        st.info(f"Your predicted house price: ${st.session_state['last_prediction']:,.0f}")

        user_question = st.text_input("Ask about your house price")

        if st.button("Ask"):
            if user_question.strip() == "":
                st.warning("Please type your question first.")
            else:
                with st.spinner("Thinking..."):
                    answer = ask_assistant(
                        user_question,
                        st.session_state['last_prediction'],
                        st.session_state['lower'],
                        st.session_state['upper'],
                        st.session_state['user_inputs']
                    )
                st.write(answer)
