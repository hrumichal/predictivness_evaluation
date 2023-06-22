import streamlit as st
from src.helpers import list_schemas, list_tables, list_columns
from src.evaluation import evaluate

def main():
    st.title("Predictiveness Evaluation")
    
    # UI components
    schemas = list_schemas()
    schema_name = st.selectbox("Schema", schemas)

    tables = list_tables(schema_name)
    table_name = st.selectbox("Table",tables)

    columns = list_columns(schema_name, table_name)
    enable_dropdown = st.checkbox("I want to provide Date Column")

    if enable_dropdown:
        date_column_name = st.selectbox("Date Column", columns)
    else:
        date_column_name = ''

    target_column_name = st.selectbox("Target Column", columns)
    
    solve_missing_action = st.selectbox("Solve Missing Action", 
                                        ("none", "replaceAll", "replaceNumeric", "replaceCategorical", "drop"))
    skip_gradient_boosting = st.checkbox("Skip Gradient Boosting")
    
    # Create a button
    button_clicked = st.button('Calculate')

    # Check if the button is clicked
    if button_clicked:
        evaluate(schema_name, table_name, date_column_name, target_column_name, solve_missing_action, skip_gradient_boosting)
    

if __name__ == "__main__":
    main()