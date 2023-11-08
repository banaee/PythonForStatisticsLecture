import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_diabetes
from scipy.stats import ttest_ind, chi2_contingency
from scipy import stats
import plotly.express as px


# Set page configuration to widen the layout
# st.set_page_config(layout="wide")


# ################################################################3
# Available datasets as Pandas dataframes
# ################################################################3

@st.cache_data
def load_sklearn_df(loaded_dataset, target_col_name, target_num=True):
    df = pd.DataFrame(loaded_dataset.data, columns=loaded_dataset.feature_names)
    if target_num:
        df[target_col_name] = loaded_dataset.target
    else:    
        df[target_col_name] = loaded_dataset.target_names[loaded_dataset.target]
    return df

@st.cache_data
def load_sns_df(dataset_name, target_col_name, cut_columns=None, bins=2):
    df = sns.load_dataset(dataset_name)
    if cut_columns:
        df[target_col_name] = pd.cut(df[target_col_name], bins=bins, labels=cut_columns)
    else:
        df = df.rename(columns={target_col_name: target_col_name})
    return df


# iris_df = load_sklearn_df(load_iris(), 'class')
wine_df = load_sklearn_df(load_wine(), 'class')
# diabetes_df = load_sklearn_df(load_diabetes(), 'class')
penguins_df = load_sns_df('penguins', 'species')
diamonds_df = load_sns_df('diamonds', 'price', ['low', 'medium', 'high'], bins=3)
diamonds_df = load_sns_df('diamonds', 'clarity')
titanic_df = load_sns_df('titanic', 'survived')
iris_df = load_sns_df('iris', 'species')
tips_df = load_sns_df('tips', 'tip', ['low', 'medium', 'high'], bins=3)
flights_df = load_sns_df('flights', None)
planets_df = load_sns_df('planets', 'method')
car_crashes_df = load_sns_df('car_crashes', 'abbrev')

# Dictionary with dataset names as keys and their respective dataframes
available_datasets = {
    'Iris Dataset': iris_df,
    'Penguins Dataset': penguins_df,
    'Diamonds Dataset': diamonds_df,
    'Wine Dataset': wine_df,
    'Titanic Dataset': titanic_df,
    'Flights Dataset': flights_df,
    'Diamonds Dataset': diamonds_df,
    'Planets Dataset': planets_df,
    'Car Crashes Dataset': car_crashes_df,
    'Tips Dataset': tips_df,
    # 'Diabetes Dataset': diabetes_df,
}




# ################################################################3

# ################################################################3

def app():


    # Title and description
    st.title('Statistical Analysis Tool for Various Datasets')
    st.write('Perform statistical analysis and hypothesis tests using different datasets')

    # Dropdown to select dataset
    selected_dataset = st.sidebar.selectbox('Select Dataset:', list(available_datasets.keys()))

    # Selected dataset
    df = available_datasets[selected_dataset]
    st.session_state.dataset = df

    # Select target column
    target_column = st.sidebar.selectbox('Select Target Column (Color):', [None]+df.columns.tolist())


    # Display a sample of the selected dataset
    # st.write(f'Sample of the {selected_dataset}:')
    # st.write(dataset.head())
    st.subheader(f'The selected dataset: {selected_dataset}')
    st.dataframe(df)
    
    class_olor = None
    if target_column: 
        class_olor = target_column
        df[target_column] = df[target_column].astype(str)

    # pr = dataset.profile_report()
    # st_profile_report(pr)

    # Select numeric columns and categorical columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # st.write('Numeric columns:', numeric_cols)
    # st.write('Categorical columns:', categorical_cols)
    df_num = df[numeric_cols]
    if df_num.isnull: df_num.dropna(inplace=True)

   
    # ################################################################3
    # Summary statistics
    # ################################################################3
    st.subheader('Summary Statistics (numeric columns):')
    st.write('Descriptive Statistics:')
    st.dataframe(df_num.describe().T)

    # selected_column1 = st.sidebar.selectbox('Select a Column:', df.columns, key="c1")
    # selected_column2 = st.sidebar.selectbox('Select a Column:', df.columns, key="c2")

    # ################################################################3
    # Plotly Visualizations
    # ################################################################3
    st.header('Data Visualizations')

    vis_plot = st.radio('Show Plotly Visualizations', ('histogram', 'scatter plot', 'scatter matrix', 'box plot', 'correlation heatmap'), key="plotly", horizontal=True)
    st.subheader(f'The selected plot: {vis_plot}')

    available_columns = df.columns.tolist() if target_column is None else list(df.columns)
    if target_column is not None and target_column in available_columns:
        available_columns.remove(target_column)

    if vis_plot == 'histogram':
        # Histogram
        selected_column = st.selectbox(
                "select a variable for Histogram", available_columns) # numeric_cols)

        hist_fig = px.histogram(df, x=selected_column, marginal='violin', color=class_olor, barmode='overlay', opacity=0.8)
        myplot(hist_fig, t=f'histogram of {selected_column}')

    elif vis_plot == 'scatter plot':
        # Scatter plot
        selected_columns = st.multiselect(
                "select **two** variables for Scatter", available_columns, [])  #list(numeric_cols)
        if len(selected_columns) == 2:
            scatter_fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1], color=class_olor)
            myplot(scatter_fig, t= f'scatter plot of {selected_columns[0]} and {selected_columns[1]}')
        else:
            st.warning('Please select only two numeric columns')

    elif vis_plot == 'scatter matrix':
        # Scatter matrixplot
        selected_columns = st.multiselect(
                "select **any** variables for Scatter", available_columns, available_columns) 
        # df[target_label] = dataset[target_label]
        fig = px.scatter_matrix(df, dimensions=selected_columns, color=class_olor)
        myplot(fig, h=820, t= f'scatter matrix plot of all numeric columns')
    elif vis_plot == 'box plot':
        # Box plot
        selected_columns = st.multiselect(
                "select **any** variables for Scatter", list(numeric_cols), list(numeric_cols))
        fig = px.box(df_num, log_y=True)
        myplot(fig, h=820, t= f'scatter matrix plot of all numeric columns')
    elif vis_plot == 'correlation heatmap':
        df_corr = df.corr()
        fig = px.imshow(df_corr, color_continuous_scale='burg')
        myplot(fig, h=820, t= 'Correlation Heatmap')
    else:
        st.warning('Please select a plot')


    # ################################################################3
    # Hypothesis tests section
    # ################################################################3
    st.title('Hypothesis Testing Section')

    # Select columns for the test
    column_1 = st.selectbox('Select Column for Group 1:', df_num.columns , index=0)
    column_2 = st.selectbox('Select Column for Group 2:', df_num.columns, index=1)

    # Choose test type
    test_type = st.radio('Select Test:', ['T-Test', 'Chi-Square Test'])

    # Perform the selected test
    if test_type == 'T-Test':
        result = ttest_ind(df_num[column_1], df_num[column_2])
        st.write('T-Test Result:')
        st.write(result)
        # Explain the t-test
        st.write(f"T-Test Explanation for '{column_1}' and '{column_2}':")
        st.write(f"The T-test compares the means of '{column_1}' and '{column_2}'. It determines whether the means of two groups are significantly different from each other.")
        if result.pvalue < 0.05:
            st.write(f"The p-value is {result.pvalue:.4f}, indicating **a statistically significant difference** between '{column_1}' and '{column_2}'.")
        else:
            st.write(f"The p-value is {result.pvalue:.4f}, suggesting **no significant difference** between '{column_1}' and '{column_2}'.")

        # Visualize the distribution of groups using a box plot
        fig = px.box(df, x=column_1, y=column_2, points="all", title='Distribution of groups using a box plot')
        st.plotly_chart(fig)



    elif test_type == 'Chi-Square Test':
        contingency_table = pd.crosstab(df[column_1], df[column_2])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        st.write('Chi-Square Test Result:')
        st.write("Chi-Square Statistic:", chi2)
        st.write("P-value:", p)
        # Explain the chi-square test
        st.write(f"Chi-Square Test Explanation for '{column_1}' and '{column_2}':")
        st.write(f"The Chi-Square test examines the relationship between '{column_1}' and '{column_2}'. It tests whether there is an association between the two categorical variables.")
        if p < 0.05:
            st.write(f"The p-value is {p:.4f}, suggesting **a significant association** between '{column_1}' and '{column_2}'.")
        else:
            st.write(f"The p-value is {p:.4f}, indicating **no significant association** between '{column_1}' and '{column_2}'.")

        # Visualize the contingency table using a heatmap
        fig = px.imshow(contingency_table.values,
                       labels=dict(x=column_2, y=column_1, color="Frequency"), title='Contingency table using a heatmap')
        st.plotly_chart(fig)


# ################################################################3
#  functions
# ################################################################3



def myplot(fig, w=820,h=640, t=''):
    fig.update_layout(
        title=t,
        width=w,
        height=h,
        plot_bgcolor='rgb(250, 250, 250)'
    )
    st.plotly_chart(fig, use_container_width=True)  



if __name__ == '__main__':
    app()
