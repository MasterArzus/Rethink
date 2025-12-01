import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import os
import glob
import yaml
import sys

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rethink.server.session_manager import SessionManager
from rethink.server.interactive import InteractiveSession, InteractiveDebugSession
from rethink.server.component import render_token_stream
from rethink.utils.visualize import generate_attention_html
from rethink.utils.config import GenerationConfig, ModelConfig, PromptConfig
from st_click_detector import click_detector
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="Rethink: LLM Debugger")

st.title("Rethink: Interactive LLM Debugging")

# --- Helper Functions ---
def load_config_files(subdir):
    path = os.path.join("/root/Rethink/configs", subdir)
    if not os.path.exists(path):
        return {}
    files = glob.glob(os.path.join(path, "*.yaml"))
    return {os.path.basename(f): f for f in files}

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

# 1. Model Config
st.sidebar.subheader("Model Configuration")
model_configs = load_config_files("models")
selected_model_config_file = st.sidebar.selectbox("Select Model Config", options=list(model_configs.keys()) + ["Custom"])

if selected_model_config_file != "Custom":
    model_cfg_path = model_configs[selected_model_config_file]
    model_cfg_data = load_yaml(model_cfg_path)
    model_path = model_cfg_data.get("model_name_or_path", "")
else:
    model_path = st.sidebar.text_input("Model Path", value="/root/autodl-fs/LLM-Research/Meta-Llama-3.1-8B-Instruct")

# 2. Generation Config
st.sidebar.subheader("Generation Configuration")
gen_configs = load_config_files("generation")
selected_gen_config_file = st.sidebar.selectbox("Select Generation Config", options=list(gen_configs.keys()) + ["Default"])

if selected_gen_config_file != "Default":
    gen_cfg_path = gen_configs[selected_gen_config_file]
    gen_cfg_data = load_yaml(gen_cfg_path)
else:
    gen_cfg_data = GenerationConfig().to_dict()

# Allow editing generation params
with st.sidebar.expander("Edit Generation Parameters"):
    temperature = st.slider("Temperature", 0.0, 2.0, float(gen_cfg_data.get("temperature", 0.4)))
    top_p = st.slider("Top P", 0.0, 1.0, float(gen_cfg_data.get("top_p", 0.7)))
    max_new_tokens = st.slider("Max New Tokens", 16, 1024, int(gen_cfg_data.get("max_new_tokens", 512)))
    
    gen_cfg_data.update({
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens
    })

# 3. Prompt Config
st.sidebar.subheader("Prompt Configuration")
prompt_configs = load_config_files("prompts")
selected_prompt_config_file = st.sidebar.selectbox("Select Prompt Config", options=list(prompt_configs.keys()) + ["Default"])

if selected_prompt_config_file != "Default":
    prompt_cfg_path = prompt_configs[selected_prompt_config_file]
    prompt_cfg_data = load_yaml(prompt_cfg_path)
else:
    prompt_cfg_data = PromptConfig().to_dict()

# --- Model Loading ---
if st.sidebar.button("Load Model"):
    with st.spinner(f"Loading model from {model_path}..."):
        try:
            model, tokenizer = SessionManager.get_resources(model_path)
            st.session_state['model'] = model
            st.session_state['tokenizer'] = tokenizer
            st.session_state['interactive_session'] = InteractiveSession(model, tokenizer)
            st.session_state['debug_session'] = InteractiveDebugSession(model, tokenizer)
            st.success("Model loaded!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

# --- Main Interface ---
if 'model' in st.session_state:
    
    # Debug Mode Toggle
    debug_mode = st.toggle("Enable Step-by-Step Debug Mode", value=False)
    
    # Input Area
    st.header("Input Prompt")
    
    # Apply Prompt Template Logic
    system_prompt = prompt_cfg_data.get("system_prompt", "")
    user_role = prompt_cfg_data.get("user_role", "user")
    
    default_prompt = "Jack sold 48 clips to his friends in April, and then he sold half as many clips in May. How many clips did Jack sell all together in April and May?"
    user_input = st.text_area("User Input", value=default_prompt, height=100)
    
    # Construct full prompt based on template type (simplified)
    if prompt_cfg_data.get("template_type") == "chat":
        full_prompt = f"<|begin_of_text|><|start_header_id|>{prompt_cfg_data.get('system_role', 'system')}<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>{user_role}<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        full_prompt = f"{system_prompt}\n\nQuestion: {user_input}\nAnswer:"
        
    with st.expander("View Full Prompt"):
        st.code(full_prompt)

    if debug_mode:
        st.header("Debug Controller")
        debug_session = st.session_state['debug_session']
        
        col_ctrl, col_view = st.columns([0.4, 0.6])
        
        with col_ctrl:
            if st.button("Start Debugging"):
                state = debug_session.start(full_prompt)
                st.session_state['debug_state'] = state
                st.rerun()
            
            if 'debug_state' in st.session_state:
                state = st.session_state['debug_state']
                
                st.info(f"Status: {state['status']}")
                st.metric("Current Layer", f"{state['layer_idx']} / {state['total_layers']}")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("Next Layer"):
                        state = debug_session.step_layer()
                        st.session_state['debug_state'] = state
                        st.rerun()
                with c2:
                    if st.button("Finish Token"):
                        state = debug_session.finish_token()
                        st.session_state['debug_state'] = state
                        st.rerun()
                with c3:
                    if st.button("Next Token"):
                        state = debug_session.sample_and_next()
                        st.session_state['debug_state'] = state
                        st.rerun()
                        
                if st.button("Run to End (Max Length)"):
                     # Loop until finished
                     with st.spinner("Running..."):
                         while state['status'] == "running" and len(debug_session.generated_tokens) < max_new_tokens:
                             state = debug_session.sample_and_next()
                         st.session_state['debug_state'] = state
                         st.rerun()

        with col_view:
            if 'debug_state' in st.session_state:
                state = st.session_state['debug_state']
                
                st.subheader("Generated Stream")
                
                # Visualize generated tokens using the component
                history = debug_session.strategy.history
                if history:
                    token_data = []
                    for i, record in enumerate(history):
                        token_data.append({
                            "index": i,
                            "token": record.token,
                            "prob": record.prob,
                            "is_critical": False, 
                            "reason": ""
                        })
                    
                    # Render the stream
                    html_content = render_token_stream(token_data, len(token_data)-1)
                    st.components.v1.html(html_content, height=200, scrolling=True)
                else:
                    st.info("Start generation to see tokens.")
                    
                with st.expander("Raw Text"):
                    st.text(state['full_text'])
                
                st.subheader("Layer Analysis")
                
                # Use the new TrajectoryAnalysis if available
                current_trajectory = debug_session.current_trajectory
                if current_trajectory and current_trajectory.current_layer_count() > 0:
                    from rethink.analysis.trajectory_analysis import TrajectoryAnalysis
                    
                    # Get the last computed layer index
                    last_layer_idx = current_trajectory.current_layer_count() - 1
                    
                    with st.spinner("Analyzing hidden state..."):
                        analyzer = TrajectoryAnalysis(current_trajectory, st.session_state['model'], st.session_state['tokenizer'])
                        
                        # Project to vocab (Logit Lens)
                        top_tokens = analyzer.project_to_vocab(last_layer_idx, k=10)
                        
                        if top_tokens:
                            st.markdown(f"**Logit Lens (Layer {last_layer_idx})**")
                            df_tokens = pd.DataFrame(top_tokens)
                            # Rename columns for display
                            df_tokens.columns = ["Token", "Probability"]
                            st.dataframe(
                                df_tokens.style.format({"Probability": "{:.4f}"}), 
                                hide_index=True,
                                use_container_width=True
                            )
                        else:
                            st.warning("Could not project hidden state.")
                            
                        # Show Drift if we have at least 2 layers
                        if last_layer_idx > 0:
                            drifts = analyzer.compute_drift()
                            if drifts:
                                last_drift = drifts[-1]
                                st.metric("Layer Drift (Cosine Dist)", f"{last_drift['cosine_distance']:.4f}")

                        # Attention Analysis
                        attn_weights = analyzer.get_attention_data(last_layer_idx)
                        if attn_weights is not None:
                            st.markdown("#### Attention Analysis")
                            # attn_weights shape: (batch, heads, 1, seq_len)
                            # Squeeze batch and query dim: (heads, seq_len)
                            # We assume batch_size=1
                            if attn_weights.dim() == 4:
                                attn_matrix = attn_weights[0, :, 0, :].detach().cpu().numpy()
                                
                                # Create a heatmap: Heads vs Token Positions
                                fig = px.imshow(
                                    attn_matrix, 
                                    labels=dict(x="Token Position", y="Head Index", color="Attention"),
                                    title=f"Layer {last_layer_idx} Attention Patterns (Heads vs Context)",
                                    aspect="auto"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"Unexpected attention shape: {attn_weights.shape}")

                elif state['hidden_states'] is not None:
                    # Fallback for initial state or if trajectory is empty
                    st.write("Initial Embedding State")
                else:
                    st.write("No hidden state available.")


    else:
        # Standard Mode (Existing Logic)
        if st.button("Generate Trace"):
            session = st.session_state['interactive_session']
            with st.spinner("Generating trace..."):
                # Use the configured params
                # Update controller config
                session.controller.cfg.prompt = PromptConfig(**prompt_cfg_data)
                session.controller.cfg.generation = GenerationConfig(**gen_cfg_data)
                
                # Re-run with updated config
                trace, analysis = session.run_initial_inference(user_input, use_template=True, max_new_tokens=max_new_tokens)
                
                st.session_state['trace'] = trace
                st.session_state['analysis'] = analysis
        
        # ... Visualization Code (Same as before) ...
        if 'trace' in st.session_state:
            trace = st.session_state['trace']
            analysis = st.session_state['analysis']
            
            st.header("Trace Visualization")
            col_trace, col_analysis = st.columns([0.7, 0.3])
            
            selected_idx = st.session_state.get('selected_token_index', 0)

            with col_trace:
                st.subheader("Token Stream")
                token_data = []
                for i, token in enumerate(trace.tokenlist):
                    is_critical = False
                    reason = ""
                    for interval in analysis:
                        if interval.start <= i <= interval.end:
                            is_critical = True
                            reason = interval.type
                            break
                    token_data.append({
                        "index": i, "token": token.token, "prob": token.prob,
                        "is_critical": is_critical, "reason": reason
                    })

                html_content = render_token_stream(token_data, selected_idx)
                clicked_id = click_detector(html_content)
                if clicked_id:
                    try:
                        new_idx = int(clicked_id.split("_")[1])
                        st.session_state['selected_token_index'] = new_idx
                        st.rerun()
                    except: pass
                
                st.divider()
                st.subheader("Global Metrics")
                df = pd.DataFrame(token_data)
                fig_prob = px.line(df, x="index", y="prob", title="Token Probability")
                fig_prob.add_vline(x=selected_idx, line_dash="dash", line_color="red")
                st.plotly_chart(fig_prob, use_container_width=True)

            with col_analysis:
                st.header("Analysis Panel")
                if 0 <= selected_idx < len(token_data):
                    selected_item = token_data[selected_idx]
                    with st.container(border=True):
                        st.subheader(f"Token: '{selected_item['token']}'")
                        st.metric("Probability", f"{selected_item['prob']:.4f}")
                        if selected_item['is_critical']:
                            st.error(f"Critical: {selected_item['reason']}")
                        
                        if 'analysis_obj' not in st.session_state:
                             from rethink.analysis.trace_analysis import TraceAnalysis
                             st.session_state['analysis_obj'] = TraceAnalysis(trace, session.controller.model, session.controller.tokenizer)
                        
                        analyzer = st.session_state['analysis_obj']
                        with st.spinner("Decoding alternatives..."):
                            alts = analyzer.get_token_alternatives(selected_idx, k=10)
                        if alts:
                            st.markdown("**Top Alternatives:**")
                            if 'top_k' in alts:
                                alt_df = pd.DataFrame(alts['top_k'], columns=["Token", "Prob"])
                                st.dataframe(alt_df.style.format({"Prob": "{:.4f}"}), hide_index=True, use_container_width=True)
                else:
                    st.info("Select a token.")

else:
    st.info("Please load a model to begin.")
