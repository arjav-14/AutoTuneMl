# # import streamlit as st
# # import requests
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # API = "http://127.0.0.1:8000"

# # st.set_page_config(layout="wide")

# # st.title("🚀 AutoML Dashboard")


# # # =============================
# # # DATASET UPLOAD
# # # =============================

# # st.header("Upload Dataset")

# # file = st.file_uploader("Upload CSV")

# # if file:

# #     response = requests.post(
# #         f"{API}/upload-dataset/",
# #         files={"file": file}
# #     )

# #     st.success("Dataset uploaded successfully")


# # # =============================
# # # SHOW DATASET
# # # =============================

# # st.header("Dataset Viewer")

# # if st.button("Show Dataset"):

# #     data = requests.get(f"{API}/show-dataset/").json()

# #     df = pd.DataFrame(data["rows"])

# #     st.dataframe(df, use_container_width=True)


# # # =============================
# # # DATASET SUMMARY
# # # =============================

# # st.header("Dataset Summary")

# # if st.button("Show Summary"):

# #     summary = requests.get(f"{API}/dataset-summary/").json()

# #     st.json(summary)


# # # =============================
# # # CORRELATION MATRIX HEATMAP
# # # =============================

# # st.header("Correlation Matrix")

# # if st.button("Show Correlation Matrix"):

# #     corr = requests.get(f"{API}/correlation-matrix/").json()

# #     df_corr = pd.DataFrame(corr["correlation_matrix"])

# #     fig, ax = plt.subplots(figsize=(8, 6))

# #     sns.heatmap(df_corr, annot=True, cmap="coolwarm")

# #     st.pyplot(fig)


# # # =============================
# # # TARGET SELECTION + TRAINING
# # # =============================

# # st.header("Train Model")

# # target = st.text_input("Enter target column")

# # if st.button("Train"):

# #     result = requests.post(
# #         f"{API}/select-target/",
# #         params={"target_column": target}
# #     )

# #     st.json(result.json())


# # # =============================
# # # LEADERBOARD
# # # =============================

# # st.header("Leaderboard")

# # if st.button("Show Leaderboard"):

# #     board = requests.get(f"{API}/leaderboard/").json()

# #     df = pd.DataFrame(board["scores"].items(),
# #                       columns=["Model", "Score"])

# #     st.dataframe(df)


# # # =============================
# # # CONFUSION MATRIX
# # # =============================

# # # =============================
# # # CONFUSION MATRIX (TP TN FP FN FORMAT)
# # # =============================

# # st.header("Confusion Matrix")

# # if st.button("Show Confusion Matrix"):

# #     response = requests.get(f"{API}/confusion-matrix/")

# #     if response.status_code != 200:
# #         st.error("Backend error while fetching confusion matrix")
# #         st.stop()

# #     data = response.json()

# #     if isinstance(data, dict) and "error" in data:
# #         st.warning(data["error"])
# #         st.stop()

# #     # Support both formats (dict OR list)
# #     if isinstance(data, dict) and "confusion_matrix" in data:
# #         cm = data["confusion_matrix"]
# #     else:
# #         cm = data


# #     df_cm = pd.DataFrame(cm)


# #     # Only valid for binary classification
# #     if df_cm.shape == (2, 2):

# #         TN = df_cm.iloc[0, 0]
# #         FP = df_cm.iloc[0, 1]
# #         FN = df_cm.iloc[1, 0]
# #         TP = df_cm.iloc[1, 1]

# #         formatted_cm = pd.DataFrame(
# #             [[TP, FN],
# #              [FP, TN]],
# #             index=["Actual Positive", "Actual Negative"],
# #             columns=["Predicted Positive", "Predicted Negative"]
# #         )

# #         st.subheader("Confusion Matrix (TP / TN / FP / FN Format)")
# #         st.table(formatted_cm)


# #         # Show metrics separately
# #         st.subheader("Breakdown")

# #         col1, col2 = st.columns(2)

# #         col1.metric("True Positive (TP)", TP)
# #         col1.metric("False Negative (FN)", FN)

# #         col2.metric("False Positive (FP)", FP)
# #         col2.metric("True Negative (TN)", TN)

# #     else:

# #         st.info("TP/TN format available only for binary classification.")


# # # =============================
# # # MODEL METRICS
# # # =============================

# # st.header("Model Metrics")

# # if st.button("Show Metrics"):

# #     metrics = requests.get(f"{API}/model-metrics/").json()

# #     st.json(metrics)


# # # =============================
# # # FEATURE IMPORTANCE
# # # =============================

# # # =============================
# # # FEATURE IMPORTANCE
# # # =============================

# # st.header("Feature Importance")

# # if st.button("Show Feature Importance"):

# #     response = requests.get(f"{API}/feature-importance/")

# #     if response.status_code != 200:
# #         st.error("Backend error while fetching feature importance")
# #         st.stop()

# #     data = response.json()

# #     # Case 1: backend returned dictionary {feature: importance}
# #     if isinstance(data, dict) and "error" not in data:

# #         df_imp = pd.DataFrame(
# #             list(data.items()),
# #             columns=["Feature", "Importance"]
# #         )

# #     # Case 2: backend returned structured ranking format
# #     elif "ranking" in data:

# #         df_imp = pd.DataFrame(
# #             data["ranking"],
# #             columns=["Feature", "Importance"]
# #         )

# #     else:

# #         st.warning("Feature importance not available for this model")
# #         st.stop()


# #     st.subheader("Feature Importance Table")
# #     st.dataframe(df_imp, use_container_width=True)


# #     st.subheader("Feature Importance Chart")

# #     fig, ax = plt.subplots()

# #     df_imp.sort_values("Importance").plot(
# #         kind="barh",
# #         x="Feature",
# #         y="Importance",
# #         ax=ax
# #     )

# #     st.pyplot(fig)


# # # =============================
# # # PREDICTION TEMPLATE
# # # =============================

# # st.header("Prediction")

# # if st.button("Load Prediction Template"):

# #     template = requests.get(
# #         f"{API}/prediction-schema/"
# #     ).json()["prediction_template"]

# #     st.session_state.template = template


# # if "template" in st.session_state:

# #     inputs = {}

# #     for k, v in st.session_state.template.items():

# #         inputs[k] = st.text_input(k, str(v))


# #     if st.button("Predict"):

# #         for k in inputs:

# #             try:
# #                 inputs[k] = float(inputs[k])
# #             except:
# #                 pass

# #         prediction = requests.post(
# #             f"{API}/predict/",
# #             json=inputs
# #         ).json()

# #         st.success(prediction)


# # # =============================
# # # SHAP EXPLANATION
# # # =============================

# # st.header("Explain Prediction")

# # if "template" in st.session_state:

# #     if st.button("Explain Prediction"):

# #         explanation = requests.post(
# #             f"{API}/explain/",
# #             json=inputs
# #         ).json()

# #         st.json(explanation)


# # # =============================
# # # DATASET HEALTH
# # # =============================

# # st.header("Dataset Health")

# # if st.button("Check Health"):

# #     health = requests.get(
# #         f"{API}/dataset-health/"
# #     ).json()

# #     st.json(health)


# # # =============================
# # # MODEL VERSION HISTORY
# # # =============================

# # st.header("Model Version History")

# # if st.button("Show Versions"):

# #     versions = requests.get(
# #         f"{API}/model-version-history/"
# #     ).json()

# #     st.json(versions)


# # # =============================
# # # DRIFT DETECTION
# # # =============================

# # st.header("Dataset Drift Detection")

# # drift_file = st.file_uploader(
# #     "Upload New Dataset for Drift Detection"
# # )

# # if drift_file:

# #     drift = requests.post(
# #         f"{API}/detect-drift/",
# #         files={"file": drift_file}
# #     ).json()

# #     st.json(drift)


# # # =============================
# # # DOWNLOAD MODEL
# # # =============================

# # st.header("Download Model")

# # if st.button("Download Model"):

# #     st.markdown(
# #         "[Download Model](http://127.0.0.1:8000/download-model/)"
# #     )


# # # =============================
# # # DOWNLOAD PIPELINE
# # # =============================

# # st.header("Download Pipeline Bundle")

# # if st.button("Download Pipeline"):

# #     st.markdown(
# #         "[Download Pipeline](http://127.0.0.1:8000/download-pipeline/)"
# #     )



# import streamlit as st
# import requests
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# API = "http://127.0.0.1:8000"

# st.set_page_config(
#     page_title="AutoML Dashboard",
#     layout="wide",
#     page_icon="🤖"
# )

# st.title("🤖 AutoML Professional Dashboard")


# # =========================
# # SIDEBAR NAVIGATION
# # =========================

# menu = st.sidebar.radio(
#     "Navigation",
#     [
#         "📂 Dataset",
#         "📊 Exploration",
#         "⚙️ Training",
#         "🏆 Leaderboard",
#         "📉 Model Diagnostics",
#         "🔍 Feature Analysis",
#         "🎯 Prediction",
#         "🧠 Explainability",
#         "📦 Pipeline & Versioning",
#         "📡 Drift Detection"
#     ]
# )


# # =========================
# # DATASET PAGE
# # =========================

# if menu == "📂 Dataset":

#     st.header("Upload Dataset")

#     file = st.file_uploader("Upload CSV")

#     if file:
#         requests.post(f"{API}/upload-dataset/", files={"file": file})
#         st.success("Dataset uploaded successfully")


#     if st.button("Show Dataset Preview"):
#         data = requests.get(f"{API}/show-dataset/").json()
#         df = pd.DataFrame(data["rows"])
#         st.dataframe(df, use_container_width=True)


# # =========================
# # DATA EXPLORATION PAGE
# # =========================

# elif menu == "📊 Exploration":

#     col1, col2 = st.columns(2)

#     with col1:
#         if st.button("Dataset Summary"):
#             summary = requests.get(f"{API}/dataset-summary/").json()
#             st.json(summary)

#     with col2:
#         if st.button("Dataset Health"):
#             health = requests.get(f"{API}/dataset-health/").json()
#             st.json(health)


#     st.subheader("Correlation Matrix")

#     if st.button("Show Correlation Matrix"):
#         corr = requests.get(f"{API}/correlation-matrix/").json()
#         df_corr = pd.DataFrame(corr["correlation_matrix"])

#         fig, ax = plt.subplots(figsize=(8, 5))
#         sns.heatmap(df_corr, annot=True, cmap="coolwarm")
#         st.pyplot(fig)


# # =========================
# # TRAIN MODEL PAGE
# # =========================

# elif menu == "⚙️ Training":

#     st.subheader("Train Model")

#     target = st.text_input("Enter Target Column")

#     if st.button("Train Model"):
#         result = requests.post(
#             f"{API}/select-target/",
#             params={"target_column": target}
#         ).json()

#         st.success("Training Complete")
#         st.json(result)


# # =========================
# # LEADERBOARD PAGE
# # =========================

# elif menu == "🏆 Leaderboard":

#     st.subheader("Model Leaderboard")

#     board = requests.get(f"{API}/leaderboard/").json()

#     df = pd.DataFrame(board["scores"].items(),
#                       columns=["Model", "Score"])

#     st.dataframe(df.sort_values("Score", ascending=False),
#                  use_container_width=True)


# # =========================
# # MODEL DIAGNOSTICS PAGE
# # =========================

# elif menu == "📉 Model Diagnostics":

#     tab1, tab2 = st.tabs(["Metrics", "Confusion Matrix"])

#     with tab1:

#         metrics = requests.get(f"{API}/model-metrics/").json()

#         cols = st.columns(len(metrics))

#         for i, (k, v) in enumerate(metrics.items()):
#             cols[i].metric(k.upper(), round(v, 4))


#     with tab2:

#         response = requests.get(f"{API}/confusion-matrix/")
#         data = response.json()

#         if isinstance(data, dict) and "error" in data:
#             st.warning(data["error"])

#         else:

#             cm = data["confusion_matrix"] if isinstance(
#                 data, dict) else data

#             df_cm = pd.DataFrame(cm)

#             if df_cm.shape == (2, 2):

#                 TN = df_cm.iloc[0, 0]
#                 FP = df_cm.iloc[0, 1]
#                 FN = df_cm.iloc[1, 0]
#                 TP = df_cm.iloc[1, 1]

#                 formatted_cm = pd.DataFrame(
#                     [[TP, FN], [FP, TN]],
#                     index=["Actual Positive",
#                            "Actual Negative"],
#                     columns=["Predicted Positive",
#                              "Predicted Negative"]
#                 )

#                 st.table(formatted_cm)


# # =========================
# # FEATURE ANALYSIS PAGE
# # =========================

# elif menu == "🔍 Feature Analysis":

#     st.subheader("Feature Importance")

#     data = requests.get(f"{API}/feature-importance/").json()

#     df_imp = pd.DataFrame(
#         list(data.items()),
#         columns=["Feature", "Importance"]
#     )

#     st.dataframe(df_imp, use_container_width=True)

#     fig, ax = plt.subplots()

#     df_imp.sort_values("Importance").plot(
#         kind="barh",
#         x="Feature",
#         y="Importance",
#         ax=ax
#     )

#     st.pyplot(fig)


# # =========================
# # PREDICTION PAGE
# # =========================

# elif menu == "🎯 Prediction":

#     template = requests.get(
#         f"{API}/prediction-schema/"
#     ).json()["prediction_template"]

#     inputs = {}

#     for k, v in template.items():
#         inputs[k] = st.text_input(k, str(v))

#     if st.button("Predict"):

#         for k in inputs:
#             try:
#                 inputs[k] = float(inputs[k])
#             except:
#                 pass

#         prediction = requests.post(
#             f"{API}/predict/",
#             json=inputs
#         ).json()

#         st.success(prediction)


# # =========================
# # EXPLAINABILITY PAGE
# # =========================

# st.subheader("Explain Prediction")

# schema = requests.get(f"{API}/prediction-schema/").json()

# if "prediction_template" in schema:

#     input_data = {}

#     for col, val in schema["prediction_template"].items():

#         if isinstance(val, str):

#             input_data[col] = st.text_input(
#                 col,
#                 val,
#                 key=f"explain_{col}"
#             )

#         else:

#             input_data[col] = st.number_input(
#                 col,
#                 float(val),
#                 key=f"explain_{col}"
#             )

#     if st.button("Explain Prediction"):

#         response = requests.post(
#             f"{API}/explain/",
#             json=input_data
#         )

#         if response.status_code == 200:

#             result = response.json()

#             st.success("Prediction Result")
#             st.write(result["prediction"])

#             explanation_df = pd.DataFrame(
#                 result["explanation"].items(),
#                 columns=["Feature", "Impact"]
#             )

#             st.subheader("Feature Contribution")

#             st.dataframe(explanation_df)

#             st.bar_chart(
#                 explanation_df.set_index("Feature")
#             )

#         else:
#             st.error("Explainability failed")

# # =========================
# # PIPELINE PAGE
# # =========================

# elif menu == "📦 Pipeline & Versioning":

#     col1, col2 = st.columns(2)

#     with col1:

#         st.subheader("Download Model")

#         st.markdown(
#             "[Download Model](http://127.0.0.1:8000/download-model/)"
#         )

#     with col2:

#         st.subheader("Download Pipeline")

#         st.markdown(
#             "[Download Pipeline](http://127.0.0.1:8000/download-pipeline/)"
#         )


#     st.subheader("Model Version History")

#     versions = requests.get(
#         f"{API}/model-version-history/"
#     ).json()

#     st.json(versions)


# # =========================
# # DRIFT DETECTION PAGE
# # =========================

# elif menu == "📡 Drift Detection":

#     drift_file = st.file_uploader(
#         "Upload Dataset For Drift Detection"
#     )

#     if drift_file:

#         drift = requests.post(
#             f"{API}/detect-drift/",
#             files={"file": drift_file}
#         ).json()

#         st.json(drift)




import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AutoML Professional Dashboard",
    layout="wide"
)

st.title("🤖 AutoML Professional Dashboard")

# =============================
# SIDEBAR NAVIGATION
# =============================

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dataset",
        "Exploration",
        "Training",
        "Leaderboard",
        "Model Diagnostics",
        "Feature Analysis",
        "Prediction",
        "Explainability",
        "Pipeline & Versioning",
        "Drift Detection"
    ]
)

# =============================
# DATASET TAB
# =============================

if menu == "Dataset":

    st.header("Upload Dataset")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:

        res = requests.post(
            f"{API}/upload-dataset/",
            files={"file": file}
        )

        if res.status_code == 200:

            st.success("Dataset uploaded successfully")

            if st.button("Show Dataset Preview"):

                preview = requests.get(
                    f"{API}/show-dataset/"
                ).json()

                df = pd.DataFrame(preview["rows"])
                st.dataframe(df)

# =============================
# EXPLORATION TAB
# =============================

elif menu == "Exploration":

    st.header("📊 Exploratory Data Analysis")

    dataset = requests.get(f"{API}/show-dataset/")

    if dataset.status_code != 200:
        st.warning("Upload dataset first")
        st.stop()

    df = pd.DataFrame(dataset.json()["rows"])

    # =============================
    # BASIC INFO
    # =============================

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.write("Column Types")

    st.dataframe(df.dtypes.astype(str))

    # =============================
    # MISSING VALUES
    # =============================

    st.subheader("Missing Values")

    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Feature", "Missing Count"]

    st.dataframe(missing_df)

    # =============================
    # NUMERIC DISTRIBUTIONS
    # =============================

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_cols) > 0:

        st.subheader("Numeric Feature Distribution")

        selected_num = st.selectbox(
            "Select numeric feature",
            numeric_cols
        )

        st.bar_chart(df[selected_num])

    # =============================
    # CATEGORICAL DISTRIBUTIONS
    # =============================

    cat_cols = df.select_dtypes(include="object").columns

    if len(cat_cols) > 0:

        st.subheader("Categorical Feature Distribution")

        selected_cat = st.selectbox(
            "Select categorical feature",
            cat_cols
        )

        st.bar_chart(df[selected_cat].value_counts())

    # =============================
    # CORRELATION MATRIX
    # =============================

    st.subheader("Correlation Matrix")

    corr = requests.get(
        f"{API}/correlation-matrix/"
    ).json()

    corr_df = pd.DataFrame(
        corr["correlation_matrix"]
    )

    st.dataframe(corr_df)

    st.bar_chart(corr_df)

    # =============================
    # TARGET RELATIONSHIP
    # =============================

    st.subheader("Feature vs Target Relationship")

    target = st.text_input(
        "Enter target column for relationship analysis"
    )

    if target in df.columns:

        numeric_cols = df.select_dtypes(
            include=["int64", "float64"]
        ).columns

        feature = st.selectbox(
            "Select feature",
            numeric_cols
        )

        st.scatter_chart(
            df[[feature, target]]
        )

# =============================
# TRAINING TAB
# =============================

elif menu == "Training":

    st.header("Train Model")

    target = st.text_input("Enter target column")

    if st.button("Train Model"):

        result = requests.post(
            f"{API}/select-target/",
            params={"target_column": target}
        ).json()

        st.success("Training completed")

        st.write(result)

# =============================
# LEADERBOARD TAB
# =============================

elif menu == "Leaderboard":

    st.header("Model Leaderboard")

    data = requests.get(
        f"{API}/leaderboard/"
    ).json()

    st.dataframe(pd.DataFrame(data["scores"], index=["Score"]).T)

# =============================
# MODEL DIAGNOSTICS TAB
# =============================

elif menu == "Model Diagnostics":

    st.header("Model Metrics")

    metrics = requests.get(
        f"{API}/model-metrics/"
    )

    if metrics.status_code == 200:

        metrics = metrics.json()

        st.json(metrics)

    st.header("Confusion Matrix")

    cm = requests.get(
        f"{API}/confusion-matrix/"
    )

    if cm.status_code == 200:

        cm_data = cm.json()

        if "confusion_matrix" in cm_data:

            df_cm = pd.DataFrame(
                cm_data["confusion_matrix"]
            )

            st.table(df_cm)

        else:

            st.info("Confusion matrix available only for classification")

# =============================
# FEATURE ANALYSIS TAB
# =============================

elif menu == "Feature Analysis":

    st.header("Feature Importance")

    imp = requests.get(
        f"{API}/feature-importance/"
    )

    if imp.status_code == 200:

        imp_data = imp.json()

        df_imp = pd.DataFrame(
            imp_data.items(),
            columns=["Feature", "Importance"]
        )

        st.dataframe(df_imp)

        st.bar_chart(
            df_imp.set_index("Feature")
        )

    else:

        st.warning("Feature importance not available")

# =============================
# PREDICTION TAB
# =============================

elif menu == "Prediction":

    st.header("Make Prediction")

    schema = requests.get(
        f"{API}/prediction-schema/"
    ).json()

    if "prediction_template" in schema:

        input_data = {}

        for col, val in schema["prediction_template"].items():

            if isinstance(val, str):

                input_data[col] = st.text_input(
                    col,
                    val,
                    key=f"predict_{col}"
                )

            else:

                input_data[col] = st.number_input(
                    col,
                    float(val),
                    key=f"predict_{col}"
                )

        if st.button("Predict"):

            pred = requests.post(
                f"{API}/predict/",
                json=input_data
            )

            if pred.status_code == 200:

                st.success("Prediction Result")

                st.write(pred.json()["prediction"])

# =============================
# EXPLAINABILITY TAB
# =============================

elif menu == "Explainability":

    st.header("Explain Prediction")

    schema = requests.get(
        f"{API}/prediction-schema/"
    ).json()

    if "prediction_template" in schema:

        explain_input = {}

        for col, val in schema["prediction_template"].items():

            if isinstance(val, str):

                explain_input[col] = st.text_input(
                    col,
                    val,
                    key=f"explain_{col}"
                )

            else:

                explain_input[col] = st.number_input(
                    col,
                    float(val),
                    key=f"explain_{col}"
                )

        if st.button("Explain Prediction"):

            res = requests.post(
                f"{API}/explain/",
                json=explain_input
            )

            if res.status_code == 200:

                result = res.json()

                st.success("Prediction Result")

                st.write(result["prediction"])

                df_exp = pd.DataFrame(
                    result["explanation"].items(),
                    columns=["Feature", "Impact"]
                )

                st.dataframe(df_exp)

                st.bar_chart(
                    df_exp.set_index("Feature")
                )

# =============================
# PIPELINE TAB
# =============================

elif menu == "Pipeline & Versioning":

    st.header("Model Version History")

    history = requests.get(
        f"{API}/model-version-history/"
    ).json()

    st.json(history)

    st.header("Download Model")

    st.link_button(
        "Download Model",
        f"{API}/download-model/"
    )

    st.link_button(
        "Download Pipeline",
        f"{API}/download-pipeline/"
    )

# =============================
# DRIFT DETECTION TAB
# =============================

elif menu == "Drift Detection":

    st.header("Dataset Health")

    health = requests.get(
        f"{API}/dataset-health/"
    ).json()

    st.json(health)

    st.header("Detect Drift")

    drift_file = st.file_uploader(
        "Upload New Dataset",
        type=["csv"]
    )

    if drift_file:

        drift = requests.post(
            f"{API}/detect-drift/",
            files={"file": drift_file}
        ).json()

        st.write(drift)