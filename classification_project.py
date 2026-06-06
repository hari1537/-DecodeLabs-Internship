import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# =====================================================================
# MACHINE LEARNING PIPELINE (The Probabilistic Engine)
# =====================================================================

# REQUIREMENT 1: Load and understand a dataset
@st.cache_data
def get_dataset_pipeline():
    # Loading the clean, built-in Iris dataset
    iris_raw = load_iris()
    
    # Organizing features into a readable table format for tracking
    feature_df = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)
    
    # Extracting matrices
    X = iris_raw.data          # Structural measurements (Features)
    y = iris_raw.target        # Flower class identifiers (Labels)
    names = iris_raw.target_names
    features = iris_raw.feature_names
    return X, y, names, features, feature_df

X, y, class_names, feature_names, df_view = get_dataset_pipeline()


# REQUIREMENT 2: Split data into training and testing sets
# Allocating 80% of data to train the AI, and holding back 20% to test it fairly
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# REQUIREMENT 3: Apply a simple classification algorithm & model training
# Initializing the supervised learning classifier
classification_model = DecisionTreeClassifier(max_depth=3, random_state=42)

# Training Phase: The algorithm discovers structural patterns automatically
classification_model.fit(X_train, y_train)

# Performance calculation against the hidden test pool
test_predictions = classification_model.predict(X_test)
calculated_accuracy = accuracy_score(y_test, test_predictions)


# =====================================================================
# INTERACTIVE STREAMLIT WEB APP LAYER (The Frontend Interface)
# =====================================================================

st.set_page_config(page_title="Data Classification Engine", page_icon="📊", layout="centered")

# App Titles
st.title("📊 Project 2: Data Classification Using AI")
st.caption("Internship Portfolio Milestone - Supervised Machine Learning Model")
st.write("---")

# Tab Layout to keep the dashboard beautifully organized
tab1, tab2, tab3 = st.tabs(["📋 1. Dataset View", "🎯 2. Model Performance", "🔮 3. Live Prediction Engine"])

with tab1:
    st.subheader("Ingested Data Profile")
    st.write("Here is a sample of the structural dataset loaded by the engine:")
    st.dataframe(df_view.head(8), use_container_width=True)
    
    st.write("### Data Handling Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Row Count", len(X))
    col2.metric("Training Set (80%)", len(X_train))
    col3.metric("Testing Set (20%)", len(X_test))

with tab2:
    st.subheader("Algorithm Evaluation Results")
    st.write("The Decision Tree classifier was tested against the evaluation data partition:")
    
    # Renders a clean success alert displaying accuracy score
    st.success(f"🏆 **Evaluation Accuracy Score:** {calculated_accuracy * 100:.2f}%")
    st.write(
        "This means the supervised algorithm accurately map-matched the structural shapes to "
        "their correct species target classifications with near-perfect confidence."
    )

with tab3:
    st.subheader("Real-Time Probabilistic Inference")
    st.write("Adjust the sliders below to see the trained AI run a real-time classification prediction:")
    
    # Interactive Sliders for users to manipulate flower morphology
    s_len = st.slider(f"🔧 {feature_names[0]}", 4.0, 8.0, 5.8, step=0.1)
    s_wid = st.slider(f"🔧 {feature_names[1]}", 2.0, 4.5, 3.0, step=0.1)
    p_len = st.slider(f"🔧 {feature_names[2]}", 1.0, 7.0, 4.3, step=0.1)
    p_wid = st.slider(f"🔧 {feature_names[3]}", 0.1, 2.5, 1.3, step=0.1)
    
    st.write("---")
    
    # Execution sequence mapping user vectors down the model tree path
    if st.button("Run Classification Model Inference", type="primary"):
        custom_query_vector = [[s_len, s_wid, p_len, p_wid]]
        predicted_index = classification_model.predict(custom_query_vector)[0]
        final_prediction = class_names[predicted_index]
        
        # Output the classification dynamically
        st.markdown(f"### 🤖 Predicted Classification: **`{final_prediction.upper()}`**")