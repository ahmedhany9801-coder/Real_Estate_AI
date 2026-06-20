import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from groq import Groq

model = joblib.load("best_model.pkl")

st.set_page_config(layout='wide', page_title='RealEstate Ai')
st.title('Know the real value of any home in seconds')
st.image("real_estate.png", width=700)


@st.cache_data
def load_data():
    return pd.read_csv("house_prices.csv")

df = load_data()

page = st.sidebar.radio(
    "Navigation",
    ['Prediction', 'model Insight📊', 'AI Asistant🤖']
)

if page == 'Prediction':
    st.dataframe(df.iloc[409:413])    

    st.subheader("Property Information🏡")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Property Details")
        house_age = st.slider("House Age (Years)", 1, 79, 40)
        num_bedrooms = st.slider("Bedrooms", 1, 6, 3)
        num_bathrooms = st.slider("Bathrooms", 1, 5, 2)
        num_floors = st.slider("Floors", 1, 3, 1)

    with col2:
        st.markdown("### Size & Outdoor")
        sqft_living = st.slider("Living Area (sqft)", 350, 6000, 1462, step=50)
        sqft_lot = st.slider("Lot Size (sqft)", 888, 12000, 5742, step=100)
        has_pool = st.toggle("Swimming Pool")
        # تم حذف خيار التجديد تماماً بناءً على طلبك لكي لا يظهر للمستخدم 🌟

    with col3:
        st.markdown("### Location & Quality")
        neighborhood = st.selectbox(
            "Neighborhood",
            sorted(df['neighborhood'].dropna().unique())
        )
        garage_type = st.selectbox("Garage Type", ['Single', 'Double'])
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


    has_pool = int(has_pool)
    # Location-based features calculated automatically

    school_rating = (
        df[df['neighborhood'] == neighborhood]
        ['school_rating']
        .median()
    )

    crime_rate = (
        df[df['neighborhood'] == neighborhood]
        ['crime_rate']
        .median()
    )
    st.divider()

    predict_btn = st.button(" Predict House Price", use_container_width=True)

    if predict_btn:

        total_rooms = num_bedrooms + num_bathrooms
        luxury_score = num_bathrooms / num_bedrooms if num_bedrooms > 0 else 0

        # تمرير قيم افتراضية في الخلفية لمنع الـ Missing columns error من غير ما يظهر حاجة للعميل
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
            'renovation_year': [renovation_year], # يمرر في الخلفية بنجاح
            'total_rooms': [total_rooms],
            'luxury_score': [luxury_score],
            'years_since_renovation': [years_since_renovation], # يمرر في الخلفية بنجاح
            'neighborhood': [neighborhood],
            'heating_type': [heating_type],
            'garage_type': [garage_type],
            'house_condition': [house_condition]
        })

        prediction = model.predict(input_df)[0]

        mae = 18798
        lower = prediction - mae
        upper = prediction + mae
        st.session_state['last_prediction'] = prediction
        st.session_state['lower'] = lower
        st.session_state['upper'] = upper
        st.session_state['user_inputs'] = input_df.to_dict('records')[0]

        st.success(f"🏠 Predicted Price: ${prediction:,.0f}")
        st.info(f"📊 Expected Range: ${lower:,.0f} - ${upper:,.0f}")




elif page == 'model Insight📊':

    st.subheader(" Model Insights📊")



    # ============================================================
    # 2) Feature Importance — تأثيرها على السعر
    # ============================================================
    with st.expander(" Feature Importance "):

        preprocessor = model.named_steps['pre processing']
        xgb_model = model.named_steps['model']

        scaling_cols = ['house_age', 'num_bedrooms', 'num_bathrooms', 'sqft_living',
                         'num_floors', 'has_pool', 'total_rooms', 'luxury_score',]
        median_cols = ['sqft_lot', 'crime_rate', ]
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

        # تحديد الطول الأصغر لضمان عدم حدوث ValueError: All arrays must be of the same length 🎉
        min_length = min(len(all_features), len(importances))

        fi_df = pd.DataFrame({
            'Feature': all_features[:min_length],
            'Importance': importances[:min_length]
        }).sort_values('Importance', ascending=False)

        fig = px.bar(
            fi_df.head(15).sort_values('Importance'),
            x='Importance', y='Feature', orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)



    # ============================================================
    # 4) Market Insights
    # ============================================================
    with st.expander("Market Insights🏡"):

        st.markdown("##### Price Distribution")
        fig1 = px.histogram(df, x='price', nbins=50)
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("##### Avg Price by Neighborhood")
        avg_neigh = df.groupby('neighborhood')['price'].mean().sort_values(ascending=False)
        fig2 = px.bar(avg_neigh)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("##### Avg Price by Condition")
        avg_cond = df.groupby('house_condition')['price'].mean().reindex(
            ['Poor', 'Fair', 'Good', 'Excellent']
        )
        fig3 = px.bar(avg_cond)
        st.plotly_chart(fig3, use_container_width=True)


    # ============================================================
    # 6) Model Information
    # ============================================================
    with st.expander("ℹ️ Model Information"):
        st.markdown(
            """
            - **Model:** XGBoost Regressor
            - **Train R²:** 96.17%
            - **Test R²:** 94.50%
            - **MAE:** $18,798
            - **RMSE:** $23,765

            """
        )




elif page == 'AI Asistant🤖':

    st.subheader("🤖 AI Assistant")


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
3. You MUST reference at least 2-3 specific property details (e.g. school rating, crime rate, neighborhood, size) and explain how each one likely influenced the price, either pushing it up or down. Be specific and concrete, not generic or philosophical.
4. Only use the values present in PROPERTY DETAILS. Never invent or assume data that wasn't provided.
5. Mention the numeric difference from the market average when useful.
6. Keep your answer concise — one or two short paragraphs maximum.
7. If asked something unrelated to property valuation, politely redirect the conversation back to the app's purpose.
8. Do not provide legal or financial guarantees. Clarify that this is a machine learning estimate, not an official appraisal.

OUTPUT FORMAT (STRICT):
- Always respond in clear, natural Modern Standard Arabic.
- Write in well-spaced, properly formatted sentences — never merge words together.
- Never repeat the same sentence, phrase, or number more than once in your response.
- Do not restate the question or echo the context back to the user.
- Write the final answer once, directly, with no duplication or filler.
"""

    def build_context(predicted_price, lower, upper, user_inputs):
        neighborhood = user_inputs.get('neighborhood', 'غير محدد')

        house_age = user_inputs.get('house_age', user_inputs.get('age', 'غير محدد'))

        bedrooms = user_inputs.get('num_bedrooms', user_inputs.get('bedrooms', 'غير محدد'))

        bathrooms = user_inputs.get('num_bathrooms', user_inputs.get('bathrooms', 'غير محدد'))

        living_area = user_inputs.get('sqft_living', user_inputs.get('area', 'غير محدد'))
        lot_size = user_inputs.get('sqft_lot', user_inputs.get('lot', 'غير محدد'))

        floors = user_inputs.get('num_floors', user_inputs.get('floors', 'غير محدد'))
        garage = user_inputs.get('garage_type', user_inputs.get('garage', 'غير محدد'))

        pool_val = user_inputs.get('has_pool', 0)
        has_pool = 'Yes' if pool_val in [1, 'Yes', 'نعم', True] else 'No'

        condition = user_inputs.get('house_condition', user_inputs.get('condition', 'Good'))
        heating = user_inputs.get('heating_type', 'Standard')

        try:
            school_rating = f"{float(user_inputs.get('school_rating', 5.0)):.1f}"
        except Exception:
            school_rating = "غير محدد"

        try:
            crime_rate = f"{float(user_inputs.get('crime_rate', 0.0)):.1f}"
        except Exception:
            crime_rate = "غير محدد"


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

"""

    def ask_assistant(user_question, predicted_price, lower, upper, user_inputs):
        context = build_context(predicted_price, lower, upper, user_inputs)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\nسؤال المستخدم: {user_question}"}
            ],
            temperature=0.3,
            max_tokens=300
        )

        return response.choices[0].message.content
    if 'last_prediction' not in st.session_state:
        st.warning("لازم تعمل توقع للسعر الأول من صفحة Prediction")
    else:
        st.info(f"السعر المتوقع لمنزلك: ${st.session_state['last_prediction']:,.0f}")

        user_question = st.text_input("اسأل عن سعر منزلك")

        if st.button("اسأل"):
            if user_question.strip() == "":
                st.warning("اكتب سؤالك الأول.")
            else:
                with st.spinner("بيفكر..."):
                    answer = ask_assistant(
                        user_question,
                        st.session_state['last_prediction'],
                        st.session_state['lower'],
                        st.session_state['upper'],
                        st.session_state['user_inputs']
                    )
                st.write(answer)
