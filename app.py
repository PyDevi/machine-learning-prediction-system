import streamlit as st
import pandas as pd
import sweetviz as sv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score


# Title
st.title("Machine Learning Prediction System")


# Sidebar
st.sidebar.title("ML Prediction System")


# Upload CSV
file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)


if file is not None:

    # Read dataset
    data = pd.read_csv(file)

    st.sidebar.success("Dataset uploaded!")


    # Dataset information
    st.sidebar.subheader("Dataset Information")
    st.sidebar.write("Rows:", data.shape[0])
    st.sidebar.write("Columns:", data.shape[1])


    # Select target column
    target = st.sidebar.selectbox(
        "Select Target Column",
        data.columns
    )


    # X and y
    X = data.drop(columns=[target])
    y = data[target]


    # Problem type
    number = pd.to_numeric(y, errors="coerce")

    if number.isna().any():
        problem = "Classification"
    else:
        problem = "Regression"


    # Show problem type
    st.sidebar.subheader("Problem Type")
    st.sidebar.write(problem)


    # Model name
    st.sidebar.subheader("Model")

    if problem == "Classification":
        st.sidebar.write("Random Forest Classifier")
    else:
        st.sidebar.write("Random Forest Regressor")


    # Run button
    run_model = st.sidebar.button("Run Model")


    # Show dataset
    st.subheader("Dataset")
    st.dataframe(data.head())


    # Convert text columns into numbers
    X = pd.get_dummies(X)
    X = X.fillna(0)


    # Classification
    if problem == "Classification":

        st.header("Random Forest Classification")


        # Convert target text into numbers
        encoder = LabelEncoder()

        y = encoder.fit_transform(
            y.astype(str)
        )


        if run_model:

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )


            # Model
            model = RandomForestClassifier(
                random_state=42
            )


            # Train
            model.fit(
                X_train,
                y_train
            )


            # Predict
            prediction = model.predict(X_test)


            # Accuracy
            accuracy = accuracy_score(
                y_test,
                prediction
            )


            # Display accuracy
            st.subheader("Model Performance")

            st.metric(
                "Accuracy",
                f"{accuracy * 100:.2f}%"
            )


            # Results
            result = pd.DataFrame({
                "Actual": encoder.inverse_transform(y_test),
                "Predicted": encoder.inverse_transform(prediction)
            })


            st.subheader("Prediction Results")

            st.dataframe(
                result.head(10)
            )


    # Regression
    else:

        st.header("Random Forest Regression")


        if run_model:

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )


            # Model
            model = RandomForestRegressor(
                random_state=42
            )


            # Train
            model.fit(
                X_train,
                y_train
            )


            # Predict
            prediction = model.predict(X_test)


            # R2 Score
            score = r2_score(
                y_test,
                prediction
            )


            # Display score
            st.subheader("Model Performance")

            st.metric(
                "R² Score",
                f"{score:.2f}"
            )


            # Results
            result = pd.DataFrame({
                "Actual": y_test,
                "Predicted": prediction
            })


            st.subheader("Prediction Results")

            st.dataframe(
                result.head(10)
            )

    # SWEETVIZ ANALYSIS
 

    st.subheader("Sweetviz Data Analysis")

    if st.button("Generate Sweetviz Report"):

        report = sv.analyze(data)

        report.show_html(
            "sweetviz_report.html",
            open_browser=False
        )

        with open(
            "sweetviz_report.html",
            "r",
            encoding="utf-8"
        ) as f:

            html = f.read()


        st.components.v1.html(
            html,
            height=1000,
            scrolling=True
        )
    