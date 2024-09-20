import streamlit as st 
from src.train import train_models_parallel, train_and_cache_models
from src.data_analysis import plot_figure

def main():
    st.set_page_config(
        page_title="Sentiment Analysis App",  
        page_icon=":stars:",        
        layout="wide",                            
        menu_items={
            'Get Help': 'https://github.com/PUVHAM/Sentiment_Analysis.git',  
            'Report a Bug': 'mailto:phamquangvu19082005@gmail.com',        
            'About': "# Sentiment Analysis App\n"
                     "Perform sentiment analysis using multiple machine learning models."
        }
    )

    st.title(':movie_camera: Sentiment Analysis App with :blue[Machine Learning]')
    
    st.markdown("""
    Welcome to the **Sentiment Analysis App**! This app allows you to classify movie reviews as positive or negative using multiple machine learning models. 
    Feel free to explore different models and see how they perform on sentiment analysis tasks.
    """)
    
    abbreviations = {
        "Decision Tree": "dt",
        "Random Forest": "rf",
        "AdaBoost": "ada",
        "Gradient Boosting": "gb"
    }

    with st.sidebar:
        st.header("Configuration")
        
        option = st.selectbox('Choose Model Type', list(abbreviations.keys()))
        short_option = abbreviations.get(option, option)
        
        on = st.toggle("Data Analysis")
        
        train_models = st.button('Train Models')
        
        if train_models:
            with st.spinner('Training...'):
                _, results = train_models_parallel(list(abbreviations.values()))
            st.session_state.models_trained = True
            st.session_state.models_results = results  
            if st.session_state.models_trained:
                st.success('Training completed!')
                        
    input_review = st.text_input('Input a movie review to analyze sentiment: (Press enter to run)', placeholder='Such a bad movie...')
    
    example_button = st.button('Use an example', type="primary")
        
    if example_button:
        input_review = "I love this movie so much!"

    if input_review:
        if not st.session_state['models_trained']:
            st.error("You need to train the model first before making predictions!")
            
        else:
            st.subheader("Sentiment Prediction")
            model, result = train_and_cache_models(short_option)
            prediction = model.predict(input_review)
            st.write(f"**Review:** {input_review}")
            st.write(f"**Model Used:** {result['model']}")
            st.write(f"**Prediction:** {prediction} {'✅' if prediction == 'Positive' else '❌'}")
            st.write(f"**Model Accuracy:** {round(result['accuracy'], 4)}")

    if on:
        st.divider()
        plot_figure()

if __name__ == "__main__":
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
        st.session_state.models_results = {}
        
    main()