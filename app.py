import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob
import yaml
import sys
import datasets

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rethink.server.session_manager import SessionManager
from rethink.server.interactive import InteractiveSession
from rethink.server.component import render_token_stream
from rethink.utils.config import GenerationConfig, PromptConfig
from st_click_detector import click_detector

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

# 4. Dataset Config
st.sidebar.subheader("Dataset Configuration")
dataset_config = load_yaml("/root/Rethink/configs/datasets.yaml")
selected_dataset_name = st.sidebar.selectbox("Select Dataset", options=["None"] + list(dataset_config.keys()))

loaded_example_question = None

if selected_dataset_name != "None":
    dataset_path = dataset_config[selected_dataset_name]
    try:
        # Load dataset from disk
        if os.path.exists(dataset_path):
            # Check if it's a directory (Arrow format) or file
            if os.path.isdir(dataset_path):
                ds = datasets.load_from_disk(dataset_path)
            else:
                # Fallback for other formats if needed, but assuming arrow/disk for now
                st.sidebar.warning("Dataset path is not a directory. Trying load_dataset...")
                ds = datasets.load_dataset(dataset_path)
            
            # Select split
            if isinstance(ds, datasets.DatasetDict):
                split = st.sidebar.selectbox("Split", options=list(ds.keys()))
                data = ds[split]
            else:
                data = ds
                
            # Select example
            max_idx = len(data) - 1
            example_idx = st.sidebar.number_input("Example Index", min_value=0, max_value=max_idx, value=0)
            
            selected_example = data[example_idx]
            question = selected_example.get("question", "")
            answer = selected_example.get("answer", "")
            
            with st.sidebar.expander("Preview Example"):
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {answer}")
            
            if st.sidebar.button("Load Example"):
                st.session_state['user_input_area'] = question
                st.rerun()
        else:
            st.sidebar.error(f"Dataset path not found: {dataset_path}")
            
    except Exception as e:
        st.sidebar.error(f"Error loading dataset: {e}")

# --- Model Loading ---
if st.sidebar.button("Load Model"):
    try:
        with st.status("Loading model resources...", expanded=True) as status:
            st.write("Initializing Session Manager...")
            # Simulate a small delay or just show steps
            
            st.write(f"Loading model weights from {model_path}...")
            model, tokenizer = SessionManager.get_resources(model_path)
            
            st.write("Initializing interactive session...")
            st.session_state['model'] = model
            st.session_state['tokenizer'] = tokenizer
            st.session_state['interactive_session'] = InteractiveSession(model, tokenizer)
            
            status.update(label="Model loaded successfully!", state="complete", expanded=False)
            
        st.success("Ready to generate!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# --- Main Interface ---
if 'model' in st.session_state:
    session = st.session_state['interactive_session']

    # Input Area
    st.header("Input Prompt")

    # Apply Prompt Template Logic
    system_prompt = prompt_cfg_data.get("system_prompt", "")
    user_role = prompt_cfg_data.get("user_role", "user")

    default_prompt = "Jack sold 48 clips to his friends in April, and then he sold half as many clips in May. How many clips did Jack sell all together in April and May?"

    if 'user_input_area' not in st.session_state:
        st.session_state['user_input_area'] = default_prompt

    user_input = st.text_area("User Input", key="user_input_area", height=100)

    # Construct full prompt based on template type (simplified)
    if prompt_cfg_data.get("template_type") == "chat":
        full_prompt = f"<|begin_of_text|><|start_header_id|>{prompt_cfg_data.get('system_role', 'system')}<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>{user_role}<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        full_prompt = f"{system_prompt}\n\nQuestion: {user_input}\nAnswer:"

    with st.expander("View Full Prompt"):
        st.code(full_prompt)

    if st.button("Generate Trace"):
        live_stream_placeholder = st.empty()
        stream_tokens = []

        def token_stream_callback(token):
            stream_tokens.append(token)
            live_stream_placeholder.markdown("".join(stream_tokens) + "▌")

        with st.spinner("Generating trace..."):
            session.controller.cfg.prompt = PromptConfig(**prompt_cfg_data)
            session.controller.cfg.generation = GenerationConfig(**gen_cfg_data)

            trace, analysis = session.run_initial_inference(
                user_input,
                use_template=True,
                max_new_tokens=max_new_tokens,
                stream_callback=token_stream_callback,
            )

            live_stream_placeholder.markdown("".join(stream_tokens))

            st.session_state['trace'] = trace
            st.session_state['analysis'] = analysis
            st.session_state['rethink_start_index'] = -1
            st.session_state['hidden_state_cache'] = {}
            st.session_state['probe_cache'] = {}
            st.session_state['selected_layer'] = None
            st.session_state.pop('analysis_obj', None)

    if 'trace' in st.session_state:
        trace = st.session_state['trace']
        analysis = st.session_state['analysis']

        st.header("Trace Visualization")
        col_trace, col_analysis = st.columns([0.7, 0.3])

        selected_idx = st.session_state.get('selected_token_index', 0)

        with col_trace:
            st.subheader("Token Stream")
            token_data = []
            rethink_start = st.session_state.get('rethink_start_index', -1)

            for i, token in enumerate(trace.tokenlist):
                is_critical = False
                reason = ""
                for interval in analysis:
                    if interval.start <= i <= interval.end:
                        is_critical = True
                        reason = interval.type
                        break

                is_new = (rethink_start != -1 and i >= rethink_start)

                token_data.append({
                    "index": i, "token": token.token, "prob": token.prob,
                    "is_critical": is_critical, "reason": reason,
                    "is_new": is_new
                })

            html_content = render_token_stream(token_data, selected_idx)
            clicked_id = click_detector(html_content, key="token_stream_click_detector")
            if clicked_id:
                try:
                    new_idx = int(clicked_id.split("_")[1])
                    if new_idx != selected_idx:
                        st.session_state['selected_token_index'] = new_idx
                        st.rerun()
                except:
                    pass

            st.divider()
            st.subheader("Global Metrics")
            df = pd.DataFrame(token_data)
            fig_prob = px.line(df, x="index", y="prob", title="Token Probability")
            fig_prob.add_vline(x=selected_idx, line_dash="dash", line_color="red")
            st.plotly_chart(fig_prob, width='stretch')

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
                            st.dataframe(alt_df.style.format({"Prob": "{:.4f}"}), hide_index=True, width='stretch')

                            alt_options = alt_df["Token"].tolist()
                            selected_alt_token = st.selectbox("Select alternative to force:", options=alt_options)

                    st.divider()
                    st.markdown("### Hidden States")
                    hs_cache = st.session_state.setdefault('hidden_state_cache', {})
                    probe_cache = st.session_state.setdefault('probe_cache', {})

                    if selected_idx not in hs_cache:
                        with st.spinner("Computing hidden states for this token..."):
                            hs_cache[selected_idx] = session.controller.compute_hidden_states_for_step(
                                trace,
                                selected_idx,
                                layers=None,
                            )

                    token_state = hs_cache.get(selected_idx)
                    if token_state:
                        trajectory = token_state.trajectory
                        layer_ids = sorted(trajectory.to_dict().keys())
                        if layer_ids:
                            selected_layer = st.session_state.get('selected_layer')
                            if selected_layer is None or selected_layer not in layer_ids:
                                selected_layer = layer_ids[-1]
                                st.session_state['selected_layer'] = selected_layer

                            st.markdown("Select a layer to inspect:")
                            grid_size = 8
                            for row_start in range(0, len(layer_ids), grid_size):
                                cols = st.columns(min(grid_size, len(layer_ids) - row_start))
                                for offset, layer_id in enumerate(layer_ids[row_start:row_start + grid_size]):
                                    with cols[offset]:
                                        if st.button(f"L{layer_id}", key=f"layer_{selected_idx}_{layer_id}"):
                                            st.session_state['selected_layer'] = layer_id
                                            st.rerun()

                            layer_state = trajectory.get_by_layer(selected_layer)
                            if layer_state:
                                logit_lens = session.controller._decode_hidden_state(layer_state.get_value(), top_k=10)
                                lens_df = pd.DataFrame(logit_lens, columns=["Token", "Prob"])
                                st.markdown(f"**Logit Lens @ Layer {selected_layer}:**")
                                st.dataframe(lens_df.style.format({"Prob": "{:.4f}"}), hide_index=True, width='stretch')

                                probe_key = f"{selected_idx}_{selected_layer}"
                                if probe_key not in probe_cache:
                                    with st.spinner("Running explanation probe..."):
                                        probe_cache[probe_key] = session.controller.probe_state(
                                            trace,
                                            selected_idx,
                                            layer_idx=selected_layer,
                                            cached_state=layer_state,
                                            cached_trajectory=trajectory,
                                        )

                                probe_result = probe_cache.get(probe_key, {})
                                st.markdown("**State Explanation:**")
                                st.write(probe_result.get("explanation", ""))
                        else:
                            st.info("No hidden states captured for this token.")
                    else:
                        st.info("Hidden state not available for this token.")

                    st.divider()
                    st.markdown("### Intervention")

                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("Rethink from here", help="Truncate trace at this token and regenerate."):
                            with st.spinner(f"Rethinking from step {selected_idx}..."):
                                new_trace, new_analysis = session.rethink_from_step(
                                    trace,
                                    selected_idx,
                                    max_new_tokens=gen_cfg_data.get("max_new_tokens", 128)
                                )
                                st.session_state['trace'] = new_trace
                                st.session_state['analysis'] = new_analysis
                                st.session_state['selected_token_index'] = selected_idx
                                st.session_state['rethink_start_index'] = selected_idx + 1
                                st.rerun()

                    with col_btn2:
                        if alts and st.button("Rethink with Selection", help="Replace current token with selected alternative and regenerate."):
                            with st.spinner(f"Forcing '{selected_alt_token}' and rethinking..."):
                                new_trace, new_analysis = session.rethink_from_step(
                                    trace,
                                    selected_idx,
                                    max_new_tokens=gen_cfg_data.get("max_new_tokens", 128),
                                    force_token=selected_alt_token
                                )
                                st.session_state['trace'] = new_trace
                                st.session_state['analysis'] = new_analysis
                                st.session_state['selected_token_index'] = selected_idx
                                st.session_state['rethink_start_index'] = selected_idx
                                st.rerun()
            else:
                st.info("Select a token.")

else:
    st.info("Please load a model to begin.")
