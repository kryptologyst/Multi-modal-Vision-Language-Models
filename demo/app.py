"""Streamlit demo for Multi-modal Vision-Language Models."""

import os
import tempfile
from typing import Dict, List, Optional

import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from src.main import MultiModalVLM
from src.utils.config import Config, load_config
from src.utils.device import get_device


# Page configuration
st.set_page_config(
    page_title="Multi-modal Vision-Language Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .result-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_vlm_model(config_path: Optional[str] = None) -> MultiModalVLM:
    """Load VLM model with caching."""
    config = load_config(config_path)
    return MultiModalVLM(config)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Multi-modal Vision-Language Models</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["CLIP", "BLIP"],
        help="Choose the vision-language model to use"
    )
    
    # Task selection
    task = st.sidebar.selectbox(
        "Select Task",
        ["Zero-shot Classification", "Visual Question Answering", "Image-Text Similarity"],
        help="Choose the task to perform"
    )
    
    # Load model
    config_path = f"configs/{model_type.lower()}.yaml" if os.path.exists(f"configs/{model_type.lower()}.yaml") else None
    vlm = load_vlm_model(config_path)
    
    # Device info
    device = get_device()
    st.sidebar.info(f"Running on: {device}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                image_path = tmp_file.name
        
        # Text input based on task
        if task == "Zero-shot Classification":
            st.subheader("Candidate Labels")
            labels_input = st.text_area(
                "Enter candidate labels (one per line)",
                value="a photo of a cat\na photo of a dog\na picture of a person",
                help="Enter possible class labels for the image"
            )
            candidate_labels = [label.strip() for label in labels_input.split('\n') if label.strip()]
            
        elif task == "Visual Question Answering":
            st.subheader("Question")
            question = st.text_input(
                "Ask a question about the image",
                value="What is in this image?",
                help="Enter a question about the uploaded image"
            )
            
        elif task == "Image-Text Similarity":
            st.subheader("Text Description")
            text_description = st.text_area(
                "Enter a text description",
                value="A beautiful landscape with mountains and trees",
                help="Enter a text description to compare with the image"
            )
    
    with col2:
        st.header("Results")
        
        if uploaded_file is not None:
            # Process based on task
            if task == "Zero-shot Classification":
                if candidate_labels:
                    with st.spinner("Performing zero-shot classification..."):
                        results = vlm.zero_shot_classification(image_path, candidate_labels)
                    
                    # Display results
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.success(f"**Predicted Label:** {results['predicted_label']}")
                    st.success(f"**Confidence:** {results['confidence']:.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Probability distribution
                    st.subheader("Probability Distribution")
                    labels = list(results['all_probabilities'].keys())
                    probs = list(results['all_probabilities'].values())
                    
                    fig = px.bar(
                        x=labels,
                        y=probs,
                        title="Classification Probabilities",
                        labels={'x': 'Labels', 'y': 'Probability'},
                        color=probs,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed metrics
                    st.subheader("Detailed Results")
                    for label, prob in results['all_probabilities'].items():
                        st.metric(label, f"{prob:.3f}")
            
            elif task == "Visual Question Answering":
                if question:
                    with st.spinner("Answering question..."):
                        results = vlm.visual_question_answering(image_path, question)
                    
                    # Display results
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.info(f"**Question:** {results['question']}")
                    st.success(f"**Answer:** {results['answer']}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            elif task == "Image-Text Similarity":
                if text_description:
                    with st.spinner("Computing similarity..."):
                        similarity = vlm.image_text_similarity(image_path, text_description)
                    
                    # Display results
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.success(f"**Similarity Score:** {similarity:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Similarity visualization
                    st.subheader("Similarity Visualization")
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = similarity,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Similarity Score"},
                        delta = {'reference': 0.5},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.3], 'color': "lightgray"},
                                {'range': [0.3, 0.7], 'color': "gray"},
                                {'range': [0.7, 1], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.8
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Please upload an image to get started.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Multi-modal Vision-Language Models Demo | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
