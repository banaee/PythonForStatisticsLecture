import streamlit as st
import pandas as pd
import numpy as np

st.write("""
# My first app
Hello *world!*
""")

data = np.random.randn(100)
df = pd.DataFrame({'Value': data})
st.line_chart(df)

