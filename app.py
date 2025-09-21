import streamlit as st
import pandas as pd
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import plotly.express as px
import json
import re

# --- Page Configuration ---
st.set_page_config(page_title="Student Stress Dashboard", layout="wide")

# --- Custom Styling (Omitted for brevity, no changes) ---
st.markdown("""
<style>
    .report-container {
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 25px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .report-container h3 {
        color: #6a11cb; /* A nice purple */
        border-bottom: 2px solid #6a11cb;
        padding-bottom: 10px;
    }
    .report-container h4 {
        color: #333;
        margin-top: 20px;
    }
    .context-box {
        border-left: 4px solid #17a2b8; /* A nice cyan */
        padding-left: 15px;
        margin-top: 15px;
        background-color: #e2f3f5;
        border-radius: 5px;
        padding-top: 1px;
        padding-bottom: 1px;
    }
</style>
""", unsafe_allow_html=True)


# --- UI (Omitted for brevity, no changes) ---
st.title("🎓 Student Stress & Well-being Dashboard")
st.markdown("Get instant, data-grounded insights and actionable recommendations to improve student well-being.")
st.subheader("Connecting Student Well-being to Global Goals")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div style="background-color:#E5F5E0; padding:15px; border-radius:10px;">
    <h4 style="color: #333;"><span style="font-size: 2em;">💚</span> SDG 3: Good Health & Well-being</h4>
    <p style="color: #333;">Ensuring healthy lives and promoting well-being for all is crucial. High stress and poor mental health among students are direct challenges to this goal.</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="background-color:#D9E8F8; padding:15px; border-radius:10px;">
    <h4 style="color: #333;"><span style="font-size: 2em;">📚</span> SDG 4: Quality Education</h4>
    <p style="color: #333;">A student's ability to learn and thrive is compromised by stress. Addressing well-being is fundamental to providing inclusive and equitable quality education.</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

# --- Helper Functions (Omitted for brevity, no changes) ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # There seem to be duplicate column names in Stress_Dataset.csv, let's fix them
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique(): 
            cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        df.columns = cols
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Required dataset '{file_path}' not found. Please ensure both CSV files are in the same directory.")
        st.stop()

def csv_to_text_chunks(df, dataset_name):
    chunks = []
    for index, row in df.iterrows():
        row_description = ", ".join([f"'{col}' is '{row[col]}'" for col in df.columns])
        chunk = f"From the '{dataset_name}' dataset, record {index} shows: {row_description}."
        chunks.append(chunk)
    return chunks

@st.cache_resource
def create_unified_vector_store(_chunks, api_key):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        return FAISS.from_texts(_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

def get_gemini_response(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error communicating with Gemini API: {e}")
        return "Sorry, an API error occurred."

def get_query_intent(query, api_key):
    prompt = f"""
    You are an expert query classifier. Your job is to analyze the user's query to determine the PRIMARY intent and return a JSON object with the classification.
    **CRITICAL RULE:** A query that asks for data analysis (e.g., "compare," "show," "predict," "generate a decision tree") AND may mention a chart type (e.g., "bar chart," "pie chart") has a PRIMARY intent of `analyze_data`. A query is only `meta_question` if it ONLY asks about capabilities.
    **Reasoning Process:**
    1.  **Classify Intent:** Determine if the primary intent is `analyze_data`, `out_of_scope`, or `meta_question`.
    2.  **Generate Explanation (if out_of_scope):** If and only if the intent is `out_of_scope`, generate a dynamic, user-facing `reason` and `suggestion`.
    **Example Scenarios:**
    *   **User Query:** "Compare average depression scores by stress type using a bar chart."
        *   **Reasoning:** The primary intent is to "Compare average depression scores." The "bar chart" part is a secondary instruction. Therefore, the intent is `analyze_data`.
        *   **JSON Output:** `{{"intent": "analyze_data"}}`
    *   **User Query:** "Generate a decision tree to predict who will experience 'Distress'."
        *   **Reasoning:** The primary intent is to "predict who will experience 'Distress'." This is a request for data analysis.
        *   **JSON Output:** `{{"intent": "analyze_data"}}`
    *   **User Query:** "Can you make a bar chart?"
        *   **Reasoning:** This query ONLY asks about capabilities. Therefore, the intent is `meta_question`.
        *   **JSON Output:** `{{"intent": "meta_question"}}`
    *   **User Query:** "Why are the SDGs important for education?"
        *   **Reasoning:** This is a philosophical 'why' question requiring general knowledge, which is outside the scope of the data.
        *   **JSON Output:** `{{"intent": "out_of_scope", "reason": "I understand you're asking about the philosophy behind the SDGs. This query requires general knowledge about global policy, whereas this application is designed to analyze its specific dataset.", "suggestion": "To get a data-driven answer, you could ask: 'How does the data show a connection between student health and academic performance?'"}}`
    ---
    **User Query to Analyze:** "{query}"
    **Your JSON Response:**
    """
    try:
        response_text = get_gemini_response(prompt, api_key)
        if response_text.strip().startswith("```json"):
            response_text = response_text.strip()[7:-3]
        data = json.loads(response_text)
        if "intent" not in data:
            return {"intent": "unknown", "reason": "Could not determine intent."}
        return data
    except (json.JSONDecodeError, TypeError):
        return {"intent": "unknown", "reason": "Received a malformed response from the classification model."}

def add_guaranteed_context(query, df, num_samples=10):
    guaranteed_chunks = []
    if re.search(r'\bage\b', query, re.IGNORECASE):
        if 'Age' in df.columns:
            sample_df = df.sample(n=min(len(df), num_samples))
            guaranteed_chunks.extend(csv_to_text_chunks(sample_df, "guaranteed sample with Age"))
    if re.search(r'\bgender\b|\bmale\b|\bfemale\b', query, re.IGNORECASE):
        if 'Gender' in df.columns:
            sample_df = df.sample(n=min(len(df), num_samples))
            guaranteed_chunks.extend(csv_to_text_chunks(sample_df, "guaranteed sample with Gender"))
    return "\n".join(guaranteed_chunks)

# --- START OF CHART LABEL IMPROVEMENTS ---

# Central dictionary for readable names and scales for BOTH datasets
COLUMN_LABELS = {
    # From StressLevelDataset.csv
    'anxiety_level': ('Anxiety Score', 21),
    'self_esteem': ('Self-Esteem Score', 30),
    'mental_health_history': ('Mental Health History (0=No, 1=Yes)', None),
    'depression': ('Depression Score', 27),
    'headache': ('Headache Frequency', 5),
    'blood_pressure': ('Blood Pressure Category', 3),
    'sleep_quality': ('Sleep Quality', 5),
    'breathing_problem': ('Breathing Problem Severity', 5),
    'noise_level': ('Noise Level Satisfaction', 5),
    'living_conditions': ('Living Conditions Satisfaction', 5),
    'safety': ('Safety Perception', 5),
    'basic_needs': ('Basic Needs Fulfillment', 5),
    'academic_performance': ('Academic Performance Rating', 5),
    'study_load': ('Study Load Perception', 5),
    'teacher_student_relationship': ('Teacher-Student Relationship', 5),
    'future_career_concerns': ('Future Career Concerns', 5),
    'social_support': ('Social Support Level', 3),
    'peer_pressure': ('Peer Pressure', 5),
    'extracurricular_activities': ('Extracurricular Activities Impact', 5),
    'bullying': ('Bullying Experience', 5),
    'stress_level': ('Primary Stress Level', 2),

    # From Stress_Dataset.csv (using cleaned names)
    'Gender': ('Gender', None),
    'Age': ('Age', None),
    'Stress': ('Recent Stress Experience', 5),
    'Heartbeat': ('Rapid Heartbeat Experience', 5),
    'anxiety': ('Recent Anxiety/Tension', 5),
    'Sleep': ('Sleep Problems', 5),
    'Concentration': ('Academic Concentration Difficulty', 5),
    'sadness': ('Sadness or Low Mood', 5),
    'irritation': ('Irritation Frequency', 5),
    'isolation': ('Loneliness or Isolation', 5),
    'Have you been feeling sadness or low mood?.1': ('Sadness or Low Mood (2)', 5),
    'Have you been experiencing any illness or health issues?.1': ('Recent Health Issues', 5),
    'Do you often feel lonely or isolated?.1': ('Loneliness or Isolation (2)', 5),
    'Do you feel overwhelmed with your academic workload?.1': ('Overwhelmed by Workload', 5),
    'Are you in competition with your peers, and does it affect you?.1': ('Affected by Peer Competition', 5),
    'Do you find that your relationship often causes you stress?.1': ('Relationship Stress', 5),
    'Are you facing any difficulties with your professors or instructors?.1': ('Difficulties with Professors', 5),
    'Is your working environment unpleasant or stressful?.1': ('Unpleasant Work Environment', 5),
    'Do you struggle to find time for relaxation and leisure activities?.1': ('Difficulty Finding Leisure Time', 5),
    'Is your hostel or home environment causing you difficulties?.1': ('Difficult Home Environment', 5),
    'Do you lack confidence in your academic performance?.1': ('Lacking Academic Confidence', 5),
    'Do you lack confidence in your choice of academic subjects?.1': ('Lacking Confidence in Subject Choice', 5),
    'Academic and extracurricular activities conflicting for you?.1': ('Conflict between Academics and ECs', 5),
    'Do you attend classes regularly?.1': ('Class Attendance Regularity', 5),
    'Have you gained/lost weight?.1': ('Recent Weight Change', 5),
    'Which type of stress do you primarily experience?': ('Primary Stress Type', None)
}

def get_label_and_scale(column_name):
    """Helper function to get a clean label and scale for a column."""
    return COLUMN_LABELS.get(column_name, (column_name.replace('_', ' ').title(), None))

# --- Chart Functions ---
def generate_pie_chart(df):
    st.write("#### Pie Chart: Proportional Distribution")
    
    # Create a mapping from pretty names to raw column names for the selectbox
    categorical_cols_raw = df.select_dtypes(exclude=['number', 'bool']).columns.tolist()
    for col in df.select_dtypes(include=['number']).columns:
        if 5 < df[col].nunique() < 10 and col not in categorical_cols_raw:
             categorical_cols_raw.append(col)
    
    if not categorical_cols_raw:
        st.warning("No suitable categorical columns found for a pie chart.")
        return
        
    pretty_to_raw_map = {get_label_and_scale(col): col for col in categorical_cols_raw}
    selected_pretty_name = st.selectbox("Select a category for the pie chart:", pretty_to_raw_map.keys(), key=f"pie_{df.attrs.get('name')}")
    
    if selected_pretty_name:
        selected_col_raw = pretty_to_raw_map[selected_pretty_name]
        pretty_title_name, _ = get_label_and_scale(selected_col_raw)
        
        counts = df[selected_col_raw].value_counts()
        fig = px.pie(values=counts.values, names=counts.index,
                     title=f'Proportional Distribution of {pretty_title_name}',
                     hole=0.4, color_discrete_sequence=px.colors.sequential.Plasma_r)
        fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05] * len(counts.index))
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def generate_bar_chart(df):
    st.write("#### Bar Chart: Comparison Across Categories")
    
    numeric_cols_raw = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols_raw = df.select_dtypes(exclude=['number', 'bool']).columns.tolist()
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].nunique() < 10 and col not in categorical_cols_raw:
             categorical_cols_raw.append(col)
             
    if not categorical_cols_raw or not numeric_cols_raw:
        st.warning("A bar chart requires at least one categorical and one numeric column.")
        return
        
    # Create mappings for selectboxes
    pretty_to_raw_cat = {get_label_and_scale(col): col for col in categorical_cols_raw}
    pretty_to_raw_num = {get_label_and_scale(col): col for col in numeric_cols_raw}
    
    col1, col2 = st.columns(2)
    with col1:
        cat_pretty = st.selectbox("Select a category (X-axis):", pretty_to_raw_cat.keys(), key=f"bar_cat_{df.attrs.get('name')}")
    with col2:
        num_pretty = st.selectbox("Select a numeric value (Y-axis):", pretty_to_raw_num.keys(), key=f"bar_num_{df.attrs.get('name')}")
        
    if cat_pretty and num_pretty:
        cat_col = pretty_to_raw_cat[cat_pretty]
        num_col = pretty_to_raw_num[num_pretty]
        
        # Get pretty labels for chart titles and axes
        cat_label, _ = get_label_and_scale(cat_col)
        num_label, num_scale = get_label_and_scale(num_col)
        
        avg_data = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).reset_index()
        
        title_text = f'Average {num_label} by {cat_label}'
        if num_scale:
            title_text += f" (out of {num_scale})"
            
        fig = px.bar(avg_data, x=cat_col, y=num_col, title=title_text,
                     color=num_col, color_continuous_scale='Plasma', text_auto='.2f')
        fig.update_traces(textposition='outside')
        fig.update_xaxes(title_text=cat_label)
        fig.update_yaxes(title_text=f'Average {num_label}')
        st.plotly_chart(fig, use_container_width=True)

def generate_regression_plot(df):
    st.write("#### Interactive Regression Analysis")
    numeric_cols_raw = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols_raw) < 2:
        st.warning("Not enough numeric columns for a regression plot.")
        return

    pretty_to_raw_map = {get_label_and_scale(col): col for col in numeric_cols_raw}

    col1, col2 = st.columns(2)
    with col1:
        x_pretty = st.selectbox("Select X-axis:", pretty_to_raw_map.keys(), index=0, key=f"reg_x_{df.attrs.get('name')}")
    with col2:
        y_pretty = st.selectbox("Select Y-axis:", pretty_to_raw_map.keys(), index=1 if len(pretty_to_raw_map) > 1 else 0, key=f"reg_y_{df.attrs.get('name')}")
        
    x_axis = pretty_to_raw_map[x_pretty]
    y_axis = pretty_to_raw_map[y_pretty]

    x_label, x_scale = get_label_and_scale(x_axis)
    y_label, y_scale = get_label_and_scale(y_axis)

    title_text = f"Relationship between {x_label} and {y_label}"
    x_axis_title = f"{x_label}{f' (out of {x_scale})' if x_scale else ''}"
    y_axis_title = f"{y_label}{f' (out of {y_scale})' if y_scale else ''}"

    reg_fig = px.scatter(df, x=x_axis, y=y_axis, trendline="ols", title=title_text,
                         labels={x_axis: x_axis_title, y_axis: y_axis_title})
    st.plotly_chart(reg_fig, use_container_width=True)

def generate_decision_tree(df):
    st.write("#### Decision Tree for Prediction")
    st.info("""
    **How to Read This Chart:** This tree predicts the target variable. At each level, the data is split based on a factor. 
    Samples are sorted to the left branch if the condition is true, and to the right if false. 
    The "value" array shows how many samples in that group fall into each final category.
    """)
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    X = df_encoded.iloc[:, :-1]
    y = df_encoded.iloc[:, -1]
    st.info(f"The decision tree is predicting the target variable: **'{df.columns[-1]}'** from the **'{df.attrs.get('name')}'** dataset.")
    try:
        dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X, y) 
        fig, ax = plt.subplots(figsize=(12, 8)) 
        class_names = [str(c) for c in sorted(y.unique())]
        plot_tree(dt, feature_names=X.columns, class_names=class_names, filled=True, ax=ax, fontsize=10, rounded=True)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not generate decision tree. Error: {e}")

# --- Main Application Logic (Omitted for brevity, no changes) ---
df_levels = load_data('StressLevelDataset.csv')
df_questions = load_data('Stress_Dataset.csv')
df_levels.attrs['name'] = 'Stress Factors (Levels)'
df_questions.attrs['name'] = 'Detailed Survey (Questions)'

st.sidebar.title("Configuration")
st.sidebar.info("This app analyzes two datasets simultaneously to provide unified insights.")
st.sidebar.write("### Datasets Preview")
st.sidebar.dataframe(df_levels.head(2))
st.sidebar.dataframe(df_questions.head(2))

if 'GEMINI_API_KEY' not in st.secrets:
    st.error("Please add your Gemini API key to Streamlit secrets!")
    st.stop()

with st.spinner("🧠 Preparing unified AI context..."):
    chunks_levels = csv_to_text_chunks(df_levels, df_levels.attrs['name'])
    chunks_questions = csv_to_text_chunks(df_questions, df_questions.attrs['name'])
    combined_chunks = chunks_levels + chunks_questions
    vector_store = create_unified_vector_store(combined_chunks, st.secrets["GEMINI_API_KEY"])

st.subheader("Ask a Question About Student Well-being")
user_query = st.text_input("Enter your query here...", placeholder="e.g., What are the key stress factors for lonely students?")

with st.expander("💡 Click for popular query examples"):
    st.markdown("""
    **Profile & Insight Questions:**
    - *What's the main cause of 'Distress' for students?*
    - *Profile students with low self-esteem and suggest actions.*
    - *What are the top 3 indicators of poor academic performance?*
    - *Are students who feel lonely more likely to have sleep problems?*

    **Graph-Based Questions:**
    - *Compare average depression scores by stress type using a **bar chart**.*
    - *Show the **distribution** of students who feel overwhelmed with a **pie chart**.*
    - *Generate a **decision tree** to predict who will experience 'Distress'.*
    """)

if st.button("Generate Insights", type="primary"):
    if user_query and vector_store:
        with st.spinner("🤔 Understanding your question..."):
            intent_data = get_query_intent(user_query, st.secrets["GEMINI_API_KEY"])
            intent = intent_data.get("intent")

        if intent == "analyze_data":
            with st.spinner("📊 Analyzing data and generating your report..."):
                retrieved_docs = vector_store.similarity_search(user_query, k=25)
                retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])
                guaranteed_context = add_guaranteed_context(user_query, df_questions)
                final_context = retrieved_context + "\n" + guaranteed_context

                # The main analysis prompt is unchanged
                prompt = f"""
                You are an expert data analyst AI. Your primary goal is to provide helpful, nuanced answers by analyzing the provided `CONTEXT FROM DATASETS` to answer the `USER QUERY`. Your response MUST be a JSON object. In your response, do not refer to yourself as an AI, a model, or a chatbot. Maintain the persona of a direct data analyst providing a report.
                **YOUR CORE DIRECTIVE: BE A HELPFUL, CAVEATED EXPERT**
                1.  **YOUR PRIMARY GOAL IS TO ANALYZE.** Always attempt to answer the user's question by finding trends and correlations in the data.
                2.  **SPECIAL INSTRUCTION FOR QUERIES WITH MISSING SPECIFICS (e.g., "movies and stress"):**
                    *   If the query's general topic (e.g., stress) is relevant, but a specific subject (e.g., movies) is NOT in the `CONTEXT`, you MUST NOT use the `out_of_scope` option.
                    *   Instead, you MUST:
                        a. In the main `report` fields, analyze the **closest available proxy** from the data (e.g., for "movies," the proxy could be "extracurricular_activities" or "leisure time"). State that you are using a proxy in your observation.
                        b. Add the optional key `"BroaderContext"` to the `report` object. In this field, use general knowledge to address the user's original question. This text MUST start with a disclaimer.
                3.  **SPECIAL INSTRUCTION FOR PREDICTION/DECISION TREE QUERIES:** If the user asks for a 'decision tree' or to 'predict' a variable, your report **MUST** focus on identifying the key factors and variables that are most influential in predicting the target. Your `Headline Insight` and `Key Indicators` should reflect this predictive analysis.
                4.  **GENERATE THE JSON REPORT:** Your default action is to generate a report using the `[Data-Grounded Report JSON Template]`. Only if the user's query is COMPLETELY unrelated should you use the `[Out-of-Scope JSON Template]`.
                ---
                **[Data-Grounded Report JSON Template]**
                ```json
                {{
                  "type": "report",
                  "report": {{
                    "Headline Insight": "(A single, powerful sentence summarizing the most critical finding from your analysis of the provided data.)",
                    "Key Indicators": [
                      {{ "Indicator": "(e.g., Anxiety Level)", "Observation": "(e.g., High, with an average score of 8/10)" }}
                    ],
                    "Consolidated Profile": "(A concise, 2-3 sentence paragraph combining the grounded indicators.)",
                    "BroaderContext": "(Optional: Use this field ONLY for queries with missing specifics. Start the text with a clear disclaimer like 'While the dataset does not contain direct data on [missing subject], broader context suggests... ')",
                    "SDG Impact & Recommended Actions": {{
                      "Impact": "(Explain the impact on SDG 3 and SDG 4 based on your findings.)",
                      "DataDrivenRecommendation": "(A concrete, actionable recommendation based *specifically* on the dataset's insights.)",
                      "GeneralResources": [
                          {{
                              "resource": "Tip: (Provide a relevant, actionable tip)",
                              "description": "(A brief explanation of the tip)"
                          }},
                          {{
                              "resource": "Authoritative Source: (e.g., WHO, CDC)",
                              "link": "(Provide a real, well-known, and authoritative URL. Do not make one up.)"
                          }}
                      ]
                    }}
                  }}
                }}
                ```
                ---
                **[Out-of-Scope JSON Template]**
                ```json
                {{
                  "type": "out_of_scope",
                  "message": "I am designed to analyze data about student stress and well-being. Your question is outside of this scope.",
                  "suggestions": [
                    "What is the connection between sleep quality and academic performance?",
                    "Compare anxiety levels between male and female students."
                  ]
                }}
                ```                ---
                **CONTEXT FROM DATASETS:**
                {final_context}
                **USER QUERY:** "{user_query}"
                **YOUR JSON RESPONSE:**
                """
                
                llm_response = get_gemini_response(prompt, st.secrets["GEMINI_API_KEY"])
                
                st.divider()
                st.write("### 📝 AI-Generated Report")

                with st.container():
                    # (Dynamic Display Logic is unchanged)
                    try:
                        if llm_response.strip().startswith("```json"):
                            llm_response = llm_response.strip()[7:-3]
                        response_data = json.loads(llm_response)
                        
                        if response_data.get("type") == "report" and "report" in response_data:
                            report = response_data["report"]
                            report_html = '<div class="report-container">'
                            if report.get("Headline Insight"): report_html += f"<h3>Headline Insight</h3><p>{report['Headline Insight']}</p><hr>"
                            if report.get("Key Indicators"):
                                report_html += "<h3>Key Indicators</h3><ul>"
                                for item in report["Key Indicators"]: report_html += f"<li><strong>{item.get('Indicator', 'N/A')}:</strong> {item.get('Observation', 'N/A')}</li>"
                                report_html += "</ul><hr>"
                            if report.get("Consolidated Profile"): report_html += f"<h3>Consolidated Profile</h3><p>{report['Consolidated Profile']}</p><hr>"
                            if report.get("BroaderContext"):
                                report_html += f"<h3>Broader Context</h3><div class='context-box'><p>{report['BroaderContext']}</p></div>"
                            if report.get("SDG Impact & Recommended Actions"):
                                sdg = report["SDG Impact & Recommended Actions"]
                                report_html += "<h3>Recommendations & Resources</h3>"
                                if sdg.get("Impact"): report_html += f"<h4>Impact on Global Goals</h4><p>{sdg['Impact']}</p>"
                                if sdg.get("DataDrivenRecommendation"): report_html += f"<h4>Data-Driven Recommendation</h4><p>{sdg['DataDrivenRecommendation']}</p>"
                                if sdg.get("GeneralResources"):
                                    report_html += "<h4>General Resources</h4><ul>"
                                    for res in sdg["GeneralResources"]:
                                        if "link" in res:
                                            report_html += f"<li><strong>{res.get('resource', 'Link')}:</strong> <a href='{res['link']}' target='_blank'>{res['link']}</a></li>"
                                        else:
                                            report_html += f"<li><strong>{res.get('resource', 'Tip')}:</strong> {res.get('description', '')}</li>"
                                    report_html += "</ul>"
                            report_html += '</div>'
                            st.markdown(report_html, unsafe_allow_html=True)
                        elif response_data.get("type") == "out_of_scope":
                            st.warning(response_data.get("message", "The query is out of scope."))
                            suggestions = response_data.get("suggestions", [])
                            if suggestions:
                                st.info("Here are some questions you could ask:\n" + "\n".join([f"- *{s}*" for s in suggestions]))
                        else:
                            st.info("The AI returned a non-standard response. Displaying the raw JSON data below.")
                            st.json(response_data)
                    except json.JSONDecodeError:
                        st.error("The AI returned a response that was not in a valid JSON format. Displaying the raw text:")
                        st.code(llm_response)
                    except Exception as e:
                        st.error(f"An error occurred while displaying the report: {e}")
                        st.code(llm_response)

        elif intent == "meta_question":
            st.success("You're asking about my capabilities! I am an AI assistant designed to analyze two datasets about student stress. You can ask me to find trends, compare factors, and create profiles based on the data. The application can also visualize this data using charts if you use keywords like 'bar chart' or 'pie chart'.")

        elif intent == "out_of_scope":
            reason = intent_data.get("reason", "Your question appears to be outside the scope of the available data.")
            suggestion = intent_data.get("suggestion", "Please try rephrasing your question to be more specific to student stress factors.")
            st.warning(f"**This question could not be answered from the provided data.**")
            st.markdown(f"**Reason:** {reason}")
            st.info(f"💡 **Suggestion:** {suggestion}")
        
        else: # Fallback for unknown intent
            st.error("Sorry, I had trouble understanding the intent of your question. Could you please rephrase it?")

        if intent == "analyze_data":
            st.divider()
            st.write("### 📊 Interactive Data Visualization")
            st.info("To explore the data yourself, select a dataset below to generate a chart.")
            dataset_for_viz = st.selectbox("Select Dataset to Visualize:",(df_levels.attrs['name'], df_questions.attrs['name']))
            df_to_visualize = df_levels if dataset_for_viz == df_levels.attrs['name'] else df_questions
            query_lower = user_query.lower()
            if any(keyword in query_lower for keyword in ["pie chart", "distribution", "proportion"]):
                generate_pie_chart(df_to_visualize)
            elif any(keyword in query_lower for keyword in ["bar chart", "compare", "average", "by"]):
                generate_bar_chart(df_to_visualize)
            elif any(keyword in query_lower for keyword in ["regression", "relationship", "correlation"]):
                generate_regression_plot(df_to_visualize)
            elif "decision tree" in query_lower:
                generate_decision_tree(df_to_visualize)
            else:
                st.write("You can also generate a chart! Ask a question that includes 'bar chart' or 'pie chart', then select a dataset.")
    else:
        st.warning("Please enter a query to generate insights.")
