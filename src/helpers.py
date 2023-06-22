import streamlit as st
import pandas as pd
import numpy as np
from src.settings import snowpark_session
from snowflake.snowpark.functions import udf, col, lit, is_null, iff, initcap


@st.cache_data
def list_schemas():
    df_is_t = snowpark_session.table('INFORMATION_SCHEMA.TABLES').select(col("TABLE_SCHEMA")).distinct().collect()
    return df_is_t

@st.cache_data
def list_tables(schema_name):
    df_is_t = snowpark_session.table('INFORMATION_SCHEMA.TABLES').filter(col("TABLE_SCHEMA")==schema_name).select(col("TABLE_NAME")).distinct().collect()
    return df_is_t

@st.cache_data
def list_columns(schema_name, table_name):
    df_is_t = snowpark_session.table('INFORMATION_SCHEMA.COLUMNS').filter((col("TABLE_SCHEMA")==schema_name) & (col("TABLE_NAME")==table_name)).select(col("COLUMN_NAME")).distinct().collect()
    df_is_t.append('')
    return df_is_t