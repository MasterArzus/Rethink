import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import os

# Add the root directory to sys.path to ensure imports work
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rethink.server.session_manager import SessionManager
from rethink.server.interactive import InteractiveSession
from rethink.server.component import render_token_stream
from rethink.utils.visualize import generate_attention_html
from st_click_detector import click_detector
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="Rethink: LLM Debugger")

st.title("Rethink: Interactive LLM Debugging")

# Sidebar for Configuration
st.sidebar.header("Configuration")

# Model Selection Logic
base_model_dir = "/root/autodl-fs/LLM-Research/"
available_models = []
if os.path.exists(base_model_dir):
    available_models = [d for d in os.listdir(base_model_dir) if os.path.isdir(os.path.join(base_model_dir, d))]

# Default to Llama if available, else first one
default_index = 0
for i, m in enumerate(available_models):
    if "Llama-3.1-8B-Instruct" in m:
        default_index = i
        break

selected_model_name = st.sidebar.selectbox(
    "Select Model", 
    options=available_models + ["Custom Path"], 
    index=default_index if available_models else 0
)

if selected_model_name == "Custom Path":
    model_path = st.sidebar.text_input("Custom Model Path", value="/root/autodl-fs/LLM-Research/Meta-Llama-3.1-8B-Instruct")
else:
    model_path = os.path.join(base_model_dir, selected_model_name)

max_new_tokens = st.sidebar.slider("Max New Tokens", min_value=16, max_value=512, value=128, step=16)
use_template = st.sidebar.checkbox("Use GSM8K Template", value=True, help="Wrap input in the standard GSM8K system prompt and format.")

# Load Model
if st.sidebar.button("Load Model"):
    with st.spinner("Loading model..."):
        try:
            model, tokenizer = SessionManager.get_resources(model_path)
            st.session_state['model'] = model
            st.session_state['tokenizer'] = tokenizer
            st.session_state['interactive_session'] = InteractiveSession(model, tokenizer)
            st.success("Model loaded!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

if 'interactive_session' in st.session_state:
    session = st.session_state['interactive_session']
    
    # Input Area
    st.header("Input Prompt")
    if use_template:
        input_label = "Question (will be wrapped in template)"
        default_prompt = "How many clips did Jack sell all together in April and May, if Jack sold 48 clips to his friends in April, and then he sold half as many clips in May. "
    else:
        input_label = "Full Prompt (Raw Input)"
        default_prompt = "Question: Jack sold 48 clips to his friends in April, and then he sold half as many clips in May.  How many clips did Jack sell all together in April and May?\nAnswer:"
        
    prompt = st.text_area(input_label, value=default_prompt, height=150)
    
    if st.button("Generate Trace"):
        with st.spinner("Generating trace..."):
            trace, analysis = session.run_initial_inference(prompt, use_template=use_template, max_new_tokens=max_new_tokens)
            st.session_state['trace'] = trace
            st.session_state['analysis'] = analysis

    # Visualization Area
    if 'trace' in st.session_state:
        trace = st.session_state['trace']
        analysis = st.session_state['analysis']
        
        st.header("Trace Visualization")
        
        # Layout: Main Trace (Left) + Analysis Panel (Right)
        col_trace, col_analysis = st.columns([0.7, 0.3])
        
        # Handle selection via query params (for clickable HTML)
        # Check if a token was clicked
        # if "token_idx" in st.query_params:
        #     try:
        #         idx = int(st.query_params["token_idx"])
        #         st.session_state['selected_token_index'] = idx
        #     except:
        #         pass
            # Clear param to avoid stuck selection on refresh (optional, but good for UX)
            # st.query_params.clear() 
            # Actually, keeping it might be fine, but let's leave it to persist state.

        selected_idx = st.session_state.get('selected_token_index', 0)

        with col_trace:
            st.subheader("Token Stream")
            
            # Prepare data for visualization
            token_data = []
            for i, token in enumerate(trace.tokenlist):
                # Determine color based on analysis
                is_critical = False
                reason = ""
                for interval in analysis:
                    if interval.start <= i <= interval.end:
                        is_critical = True
                        reason = interval.type
                        break
                
                token_data.append({
                    "index": i,
                    "token": token.token,
                    "prob": token.prob,
                    "is_critical": is_critical,
                    "reason": reason
                })

            # Generate HTML for "Sentence Flow" with Clickable Cards
            html_content = render_token_stream(token_data, selected_idx)
            
            # Use click_detector instead of st.markdown
            clicked_id = click_detector(html_content)
            
            if clicked_id:
                # ID format: token_{index}
                try:
                    new_idx = int(clicked_id.split("_")[1])
                    st.session_state['selected_token_index'] = new_idx
                    st.rerun()
                except:
                    pass
            
            # Metrics Plots (Below the text)
            st.divider()
            st.subheader("Global Metrics")
            df = pd.DataFrame(token_data)
            
            tab1, tab2 = st.tabs(["Probability", "Entropy"])
            with tab1:
                fig_prob = px.line(df, x="index", y="prob", title="Token Probability")
                # Add marker for selected
                fig_prob.add_vline(x=selected_idx, line_dash="dash", line_color="red")
                st.plotly_chart(fig_prob, use_container_width=True)
            with tab2:
                # Placeholder for entropy if not computed
                st.info("Entropy computation requires full analysis.")

        with col_analysis:
            st.header("Analysis Panel")
            
            tab_details, tab_attention = st.tabs(["Token Details", "Attention"])
            
            with tab_details:
                if 0 <= selected_idx < len(token_data):
                    selected_item = token_data[selected_idx]
                    
                    # Card-like container for details
                    with st.container(border=True):
                        st.subheader(f"Token: '{selected_item['token']}'")
                        st.caption(f"Index: {selected_item['index']}")
                        
                        st.metric("Probability", f"{selected_item['prob']:.4f}")
                        
                        if selected_item['is_critical']:
                            st.error(f"Critical: {selected_item['reason']}")
                        
                        # Fetch alternatives
                        if 'analysis_obj' not in st.session_state:
                             from rethink.analysis.trace_analysis import TraceAnalysis
                             st.session_state['analysis_obj'] = TraceAnalysis(trace, session.controller.model, session.controller.tokenizer)
                        
                        analyzer = st.session_state['analysis_obj']
                        
                        with st.spinner("Decoding alternatives..."):
                            alts = analyzer.get_token_alternatives(selected_idx, k=10)
                        
                        if alts:
                            st.metric("Entropy", f"{alts.get('entropy', 0.0):.4f}")
                            st.markdown("**Top Alternatives:**")
                            if 'top_k' in alts:
                                alt_df = pd.DataFrame(alts['top_k'], columns=["Token", "Prob"])
                                st.dataframe(
                                    alt_df.style.format({"Prob": "{:.4f}"}), 
                                    hide_index=True,
                                    use_container_width=True
                                )
                    
                    # Intervention in the side panel
                    with st.container(border=True):
                        st.subheader("Intervention")
                        st.caption(f"Edit starting from here")
                        
                        new_text = st.text_input("New Text", value="", key=f"input_{selected_idx}")
                        
                        if st.button("Apply", use_container_width=True):
                            with st.spinner("Running intervention..."):
                                new_trace, new_analysis = session.run_intervention(new_text, selected_idx)
                                st.session_state['trace'] = new_trace
                                st.session_state['analysis'] = new_analysis
                                st.rerun()
                else:
                    st.info("Select a token to view details.")
            
            with tab_attention:
                st.info("Visualize attention weights for the full sequence.")
                if st.button("Generate Attention View", use_container_width=True):
                    with st.spinner("Computing attention (this may take a moment)..."):
                        # Reconstruct full text
                        full_text = trace.question + trace.get_full_text()
                        html_code = generate_attention_html(session.controller.model, session.controller.tokenizer, full_text)
                        components.html(html_code, height=800, scrolling=True)


    # Session Management
    st.sidebar.header("Session Management")
    if st.sidebar.button("Save Current Session"):
        if 'interactive_session' in st.session_state and st.session_state['interactive_session'].current_trace:
            filepath = st.session_state['interactive_session'].save_session()
            st.sidebar.success(f"Session saved to {filepath}")
        else:
            st.sidebar.warning("No trace to save.")


else:
    st.info("Please load a model to begin.")
