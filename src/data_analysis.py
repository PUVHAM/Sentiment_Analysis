import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from src.config import DatasetConfig
from src.load_dataset import load_df

# Function to create the text inside the pie chart
def func(pct, allvalues):
    absolute = int(pct / 100. * np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)

def label_distribution_statistics(df):
    freq_pos = len(df[df['sentiment'] == 'positive'])
    freq_neg = len(df[df['sentiment'] == 'negative'])

    data = [freq_pos, freq_neg]

    # Create pie chart
    fig, ax = plt.subplots(figsize=[5, 5])
    plt.pie(x=data, autopct=lambda pct: func(pct, data), explode=[0.0025]*2,
            pctdistance=0.5, colors=[sns.color_palette()[0], 'tab:red'], textprops={'fontsize': 16})
    plt.title('Frequencies of sentiment labels', fontsize =14 ,fontweight ='bold')
    labels = [r'Positive', r'Negative']
    plt.legend(labels, loc="best", prop={'size': 14})

    return fig
    
def review_length_statistics(df):
    words_len = df['review'].str.split().map(lambda x: len(x))
    
    df_temp = df.copy()
    df_temp['words length'] = words_len
    x = "words length"
    
    fig_positive, ax_positive = plt.subplots(figsize=(5, 5))
    sns.histplot(
        data=df_temp[df_temp['sentiment'] == 'positive'],
        x=x, hue="sentiment", kde=True, ax=ax_positive
    ).set(title='Words in positive reviews') 
    
    fig_negative, ax_negative = plt.subplots(figsize=(5, 5))
    sns.histplot(
        data=df_temp[df_temp['sentiment'] == 'negative'],
        x=x, hue="sentiment", kde=True, ax=ax_negative, palette=['red']
    ).set(title='Words in negative reviews')
    
    fig_kernel, ax_kernel = plt.subplots(figsize=(5, 5))
    sns.kdeplot(
        data=df_temp, x=x, hue="sentiment", fill=True, ax=ax_kernel, palette=[sns.color_palette()[0], 'red']
    ).set(title='Words in reviews')
    
    ax_kernel.legend(title='Sentiment', labels=['negative', 'positive'])

    return fig_positive, fig_negative, fig_kernel

def plot_figure():
    df = load_df(DatasetConfig.DATASET_PATH)
    
    st.subheader("Data Analysis")
    with st.spinner('Processing...'):
        pie_fig = label_distribution_statistics(df)
        fig_positive, fig_negative, fig_kernel = review_length_statistics(df)
    
    
    cols = st.columns(4)
    
    lst_fig = [pie_fig, fig_positive, fig_negative, fig_kernel]
    
    for i, fig in enumerate(lst_fig):
        with cols[i]:
            st.pyplot(fig)