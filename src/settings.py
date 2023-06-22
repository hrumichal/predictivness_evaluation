#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import streamlit as st

from snowflake.snowpark import Session
from snowflake.snowpark.functions import udf, col, lit, is_null, iff, initcap

# Initiate Snowpark Session
KEBOOLA_HOST = st.secrets["keboola_host"]
KEBOOLA_WAREHOUSE = st.secrets["keboola_warehouse"]
KEBOOLA_DATABASE = st.secrets["keboola_database"]
KEBOOLA_SCHEMA = st.secrets["keboola_schema"]
KEBOOLA_USER = st.secrets["keboola_user"]
KEBOOLA_PASSWORD = st.secrets["keboola_password"]
KEBOOLA_ACCOUNT = st.secrets["keboola_account"]
KEBOOLA_BACKEND = st.secrets["keboola_backend"]

connection_parameters = {'host': KEBOOLA_HOST,
                         'warehouse': KEBOOLA_WAREHOUSE,
                         'database': KEBOOLA_DATABASE,
                         'schema': KEBOOLA_SCHEMA,
                         'user': KEBOOLA_USER,
                         'password': KEBOOLA_PASSWORD,
                         'account': KEBOOLA_ACCOUNT,
                         'backend': KEBOOLA_BACKEND}

snowpark_session = Session.builder.configs(connection_parameters).create()    
