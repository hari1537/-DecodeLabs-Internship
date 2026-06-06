import streamlit as st
import pandas as pd

# =====================================================================
# 🛠️ PHASE 1: DATA HANDLING & RESOURCE LIBRARY
# =====================================================================

@st.cache_data
def get_advanced_content_library():
    """Sets up a multi-genre structured catalog matrix with binary feature weights."""
    catalog = [
        # MOVIES
        {"Title": "Inception", "Type": "Movie", "Action": 1, "Sci-Fi": 1, "Drama": 1, "Comedy": 0, "Docu": 0, "Desc": "A mind-bending heist thriller through the architecture of dreams."},
        {"Title": "The Dark Knight", "Type": "Movie", "Action": 1, "Sci-Fi": 0, "Drama": 1, "Comedy": 0, "Docu": 0, "Desc": "A gritty psychological battle between justice and chaotic anarchy."},
        {"Title": "Interstellar", "Type": "Movie", "Action": 0, "Sci-Fi": 1, "Drama": 1, "Comedy": 0, "Docu": 0, "Desc": "An epic space exploration journey evaluating love across dimensions."},
        {"Title": "Superbad", "Type": "Movie", "Action": 0, "Sci-Fi": 0, "Drama": 0, "Comedy": 1, "Docu": 0, "Desc": "An iconic, high-energy comedy focused on high school friendships."},
        {"Title": "The Social Dilemma", "Type": "Movie", "Action": 0, "Sci-Fi": 1, "Drama": 0, "Comedy": 0, "Docu": 1, "Desc": "An eye-opening tech documentary analyzing algorithmic addiction."},
        {"Title": "Spider-Man: Across the Spider-Verse", "Type": "Movie", "Action": 1, "Sci-Fi": 1, "Drama": 0, "Comedy": 1, "Docu": 0, "Desc": "An animated sci-fi masterpiece exploring multiversal paths."},
        
        # TV SHOWS
        {"Title": "Our Planet", "Type": "TV Show", "Action": 0, "Sci-Fi": 0, "Drama": 0, "Comedy": 0, "Docu": 1, "Desc": "An awe-inspiring docuseries highlighting Earth's fragile biomes."},
        {"Title": "The Office", "Type": "TV Show", "Action": 0, "Sci-Fi": 0, "Drama": 0, "Comedy": 1, "Docu": 0, "Desc": "A legendary mockumentary sitcom documenting office mundanity."},
        {"Title": "Black Mirror", "Type": "TV Show", "Action": 0, "Sci-Fi": 1, "Drama": 1, "Comedy": 0, "Docu": 0, "Desc": "A dark, dystopian anthology examining modern technology's pitfalls."},
        {"Title": "Formula 1: Drive to Survive", "Type": "TV Show", "Action": 1, "Sci-Fi": 0, "Drama": 1, "Comedy": 0, "Docu": 1, "Desc": "A high-octane docuseries following intense racing politics."},
        {"Title": "Rick and Morty", "Type": "TV Show", "Action": 1, "Sci-Fi": 1, "Drama": 0, "Comedy": 1, "Docu": 0, "Desc": "A chaotic animated comedy exploring cosmic sci-fi concepts."}
    ]
    return pd.DataFrame(catalog)

df_items = get_advanced_content_library()

# =====================================================================
# 🧠 PHASE 2: PATTERN MATCHING & CORE LOGIC
# =====================================================================

def calculate_similarity_rankings(user_vector, selected_type):
    """Computes real-time vector overlap scores to sort matching targets."""
    # Filter the library pool by media type choice first
    df_pool = df_items[df_items["Type"] == selected_type].copy()
    
    match_scores = []
    confidence_percentages = []
    
    # Mathematical Max Score possible for scaling calculations
    max_possible_score = sum(user_vector.values()) * 1
    
    for _, item in df_pool.iterrows():
        # Core Vector Dot-Product Calculation (Pattern Alignment)
        dot_product = (
            (user_vector["Action"] * item["Action"]) +
            (user_vector["Sci-Fi"] * item["Sci-Fi"]) +
            (user_vector["Drama"] * item["Drama"]) +
            (user_vector["Comedy"] * item["Comedy"]) +
            (user_vector["Docu"] * item["Docu"])
        )
        match_scores.append(dot_product)
        
        # Calculate a Match Confidence Percentage
        confidence = (dot_product / max_possible_score * 100) if max_possible_score > 0 else 0
        confidence_percentages.append(round(confidence, 1))
        
    df_pool["Match Score"] = match_scores
    df_pool["Confidence %"] = confidence_percentages
    
    # Arrange dataset starting with highest mathematical similarity match
    return df_pool.sort_values(by="Match Score", ascending=False)

# =====================================================================
# 🎨 PHASE 3: USER INTERFACE DISPLAY LAYER
# =====================================================================

st.set_page_config(page_title="AI Match Engine", page_icon="🍿", layout="wide")

st.title("🍿 Project 3: AI Vector-Space Recommendation Engine")
st.caption("Internship Milestone 03 - Multi-Attribute Profile Similarity Filtering")
st.write("---")

# Visual layout optimization using split columns
col_sidebar, col_main = st.columns([1, 1.3], gap="large")

with col_sidebar:
    st.subheader("👤 User Preference Input Vector")
    st.write("Construct your personalized entertainment profile weights:")
    
    # Requirement 1: User selection/choices
    media_selection = st.selectbox("🎯 Target Media Categorization", ["Movie", "TV Show"])
    
    st.write("---")
    st.write("💡 Set your affinity levels (0 = Skip, 5 = High Priority):")
    v_action = st.slider("💥 Action & Adventure", 0, 5, 4)
    v_scifi = st.slider("🚀 Science Fiction / Tech", 0, 5, 5)
    v_drama = st.slider("🎭 Emotional Drama", 0, 5, 2)
    v_comedy = st.slider("😂 Sitcom / Comedy", 0, 5, 0)
    v_docu = st.slider("🌿 Real-world Documentaries", 0, 5, 0)
    
    # Package inputs into a unified feature profile vector
    active_user_vector = {
        "Action": v_action, "Sci-Fi": v_scifi, "Drama": v_drama,
        "Comedy": v_comedy, "Docu": v_docu
    }

with col_main:
    st.subheader("🤖 Top Algorithmic Recommendations")
    st.write("Computing vector distances to retrieve closest pattern profiles:")
    
    # Requirement 2 & 3: Match and display items dynamically
    df_recommendations = calculate_similarity_rankings(active_user_vector, media_selection)
    
    # Grab top 3 matches where score > 0
    top_matches = df_recommendations[df_recommendations["Match Score"] > 0].head(3)
    
    if not top_matches.empty:
        for idx, row in top_matches.iterrows():
            # Build a premium card interface container for each item
            with st.container(border=True):
                col_title, col_metric = st.columns([2, 1])
                with col_title:
                    st.markdown(f"### {row['Title']}")
                    st.write(row["Desc"])
                with col_metric:
                    st.metric(label="Match Confidence", value=f"{row['Confidence %']}%")
                
                # Visual match intensity progress bar
                st.progress(row["Confidence %"] / 100)
    else:
        st.info("Increase your interest affinity sliders on the left to activate pattern-matching calculations!")
        
    # Expandable catalog view section for presentation transparency
    with st.expander("📊 View Underlying Library Feature Matrix"):
        st.write("This is the 'White Box' source catalog the engine parses vectors against:")
        st.dataframe(df_items, use_container_width=True)