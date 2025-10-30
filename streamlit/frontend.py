import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.representations import extract_representations
import tempfile
import os
import numpy as np
import joblib

st.title("Fine-tuning Data Filter")

# Model loading (fixed for prototype)
@st.cache_resource
def load_classifier():
    return joblib.load("classifiers/pca_LR.pkl")

# File upload
uploaded_file = st.file_uploader("Upload your fine-tuning data (CSV)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"📊 Loaded {len(df)} examples")
    st.dataframe(df.head())
    
    # Column configuration
    st.subheader("Configure Data Columns")
    st.caption("Select which columns your data contains")
    
    columns = df.columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Checkboxes for column types
        has_instruction = st.checkbox("Instruction column")
        has_input = st.checkbox("Input column")
        has_query = st.checkbox("Query column")
        has_response = st.checkbox("Response column")
    
    with col2:
        # Show dropdowns only for checked items
        instruction_col = None
        input_col = None
        query_col = None
        response_col = None
        
        if has_instruction:
            instruction_col = st.selectbox("Select instruction column", columns, key="instruction")
        
        if has_input:
            input_col = st.selectbox("Select input column", columns, key="input")
        
        if has_query:
            query_col = st.selectbox("Select query column", columns, key="query")
        
        if has_response:
            response_col = st.selectbox("Select response column", columns, key="response")
    
    # Only query checkbox
    only_query = st.checkbox(
        "Extract from query/instruction only",
        value=True,
        help="If unchecked, extracts from full sequence (query + response)"
    )
    
    # Validation
    if not (has_instruction or has_query):
        st.warning("⚠️ Please select at least instruction OR query column")
    
    # Filter button
    elif st.button("🔍 Filter Data", type="primary"):
        with st.spinner("Processing..."):
            # Create temp files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_input:
                df.to_csv(tmp_input.name, index=False)
                input_path = tmp_input.name
            
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_output:
                output_path = tmp_output.name
            
            try:
                # Extract representations
                st.write("⚙️ Generating embeddings...")
                extract_representations(
                    model_path="models/Llama-2-7b-chat-hf",  # Your fixed model path
                    input_path=input_path,
                    output_path=output_path,
                    instruction_column=instruction_col,
                    input_column=input_col,
                    query_column=query_col,
                    response_column=response_col,
                    batch_size=32,
                    layer=-1,
                    device='cuda',
                    only_query=only_query,
                    save_index=True
                )
                
                # Load representations
                representations = np.load(output_path)
                
                # Classify
                st.write("🤖 Classifying safety...")
                classifier = load_classifier()
                predictions = classifier.predict(representations)
                
                # Filter safe examples (assuming 1 = safe, 0 = unsafe)
                df['is_safe'] = predictions
                safe_df = df[df['is_safe'] == 1].drop(columns=['is_safe'])
                unsafe_count = len(df) - len(safe_df)
                
                # Show results
                st.success("✅ Filtering complete!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Original examples", len(df))
                col2.metric("Safe examples", len(safe_df))
                col3.metric("Filtered out", unsafe_count, delta=f"-{unsafe_count}")
                
                # Preview
                st.subheader("Preview of filtered data")
                st.dataframe(safe_df.head(10))
                
                # Download button
                csv = safe_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data",
                    data=csv,
                    file_name="filtered_safe_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                
            finally:
                # Cleanup temp files
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)