import streamlit as st
import pandas as pd
import os
import math
import glob
import yaml
import sys
import datasets
import uuid
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

with st.sidebar.expander("Preview Prompt Config"):
    st.code(yaml.dump(prompt_cfg_data, default_flow_style=False), language="yaml")

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
        if hasattr(st.session_state['tokenizer'], "apply_chat_template") and st.session_state['tokenizer'].chat_template:
            messages = [
                {"role": prompt_cfg_data.get('system_role', 'system'), "content": system_prompt},
                {"role": user_role, "content": user_input}
            ]
            full_prompt = st.session_state['tokenizer'].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback to Llama 3 style if no chat template found (legacy behavior)
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
            st.session_state['selected_token_index'] = 0
            st.session_state['trace_id'] = str(uuid.uuid4())

    if 'trace' in st.session_state:
        trace = st.session_state['trace']
        analysis = st.session_state['analysis']

        st.header("Token Trace")

        selected_idx = st.session_state.get('selected_token_index', 0)

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
        trace_id = st.session_state.get('trace_id', 'default')
        clicked_id = click_detector(html_content, key=f"token_stream_click_detector_{trace_id}")
        if clicked_id:
            try:
                new_idx = int(clicked_id.split("_")[1])
                if new_idx != selected_idx:
                    st.session_state['selected_token_index'] = new_idx
                    st.rerun()
            except:
                pass

        st.divider()

        if 0 <= selected_idx < len(token_data):
            selected_item = token_data[selected_idx]

            if 'analysis_obj' not in st.session_state:
                from rethink.analysis.trace_analysis import TraceAnalysis
                st.session_state['analysis_obj'] = TraceAnalysis(trace, session.controller.model, session.controller.tokenizer)

            analyzer = st.session_state['analysis_obj']
            with st.spinner("Decoding alternatives..."):
                alts = analyzer.get_token_alternatives(selected_idx, k=10)

            alt_df = None
            alt_options = []
            if alts and 'top_k' in alts:
                alt_df = pd.DataFrame(alts['top_k'], columns=["Token", "Prob"])
                alt_options = alt_df["Token"].tolist()

            hs_cache = st.session_state.setdefault('hidden_state_cache', {})
            probe_cache = st.session_state.setdefault('probe_cache', {})
            st.session_state.setdefault('branch_candidates', [])
            st.session_state.setdefault('branch_origin_idx', None)

            tabs = st.tabs(["Overview", "Alternatives & Interventions", "Branching", "Explanations"])

            with tabs[0]:
                st.subheader(f"Token: '{selected_item['token']}'")
                st.metric("Probability", f"{selected_item['prob']:.4f}")
                if selected_item['is_critical']:
                    st.error(f"Critical: {selected_item['reason']}")
                else:
                    st.info("No critical interval flagged for this token.")

            with tabs[1]:
                st.subheader("Alternatives & Interventions")
                if alt_df is not None:
                    alt_col, action_col = st.columns(2)
                    with alt_col:
                        st.markdown("**Top Alternatives**")
                        st.dataframe(alt_df.style.format({"Prob": "{:.4f}"}), hide_index=True, width='stretch')

                    with action_col:
                        steering_prompt = st.text_input("Steering prompt (optional)", key=f"steer_{selected_idx}")
                        current_alt = st.session_state.get('selected_alt_token')
                        if not current_alt or current_alt not in alt_options:
                            current_alt = alt_options[0]
                            st.session_state['selected_alt_token'] = current_alt

                        choice = st.selectbox(
                            "Select alternative to force:",
                            options=alt_options,
                            index=alt_options.index(current_alt),
                        )
                        st.session_state['selected_alt_token'] = choice

                        st.divider()
                        st.markdown("**Interventions**")
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            if st.button("Rethink from here", help="Truncate trace at this token and regenerate."):
                                with st.spinner(f"Rethinking from step {selected_idx}..."):
                                    new_trace, new_analysis = session.rethink_from_step(
                                        trace,
                                        selected_idx,
                                        max_new_tokens=gen_cfg_data.get("max_new_tokens", 128),
                                        steering_prompt=steering_prompt,
                                    )
                                    st.session_state['trace'] = new_trace
                                    st.session_state['analysis'] = new_analysis
                                    st.session_state['selected_token_index'] = selected_idx
                                    st.session_state['rethink_start_index'] = selected_idx + 1
                                    st.rerun()

                        with col_btn2:
                            selected_alt_token = st.session_state.get('selected_alt_token')
                            if selected_alt_token:
                                if st.button("Rethink with Selection", help="Replace current token with selected alternative and regenerate."):
                                    with st.spinner(f"Forcing '{selected_alt_token}' and rethinking..."):
                                        new_trace, new_analysis = session.rethink_from_step(
                                            trace,
                                            selected_idx,
                                            max_new_tokens=gen_cfg_data.get("max_new_tokens", 128),
                                            force_token=selected_alt_token,
                                            steering_prompt=steering_prompt,
                                        )
                                        st.session_state['trace'] = new_trace
                                        st.session_state['analysis'] = new_analysis
                                        st.session_state['selected_token_index'] = selected_idx
                                        st.session_state['rethink_start_index'] = selected_idx
                                        st.rerun()
                            else:
                                st.info("Select an alternative to enable forced rethink.")
                else:
                    st.info("No alternatives available.")

            with tabs[2]:
                st.subheader("Branching")
                branch_k = st.slider("Number of branches (K)", min_value=2, max_value=5, value=3, step=1)
                branch_strategy = st.radio("Strategy", options=["sample", "beam"], horizontal=True)
                branch_steer = st.text_input("Steering prompt (optional) for branches", key=f"branch_steer_{selected_idx}")

                if st.button("Generate branches from this token"):
                    with st.spinner("Generating branches..."):
                        branches = session.branch_from_step(
                            trace,
                            selected_idx,
                            k=branch_k,
                            strategy=branch_strategy,
                            max_new_tokens=gen_cfg_data.get("max_new_tokens", 128),
                            steering_prompt=branch_steer,
                        )
                        st.session_state['branch_candidates'] = branches
                        st.session_state['branch_origin_idx'] = selected_idx

                branch_candidates = st.session_state.get('branch_candidates', [])
                if branch_candidates:
                    st.markdown("**Branch candidates:**")
                    cols = st.columns(min(3, len(branch_candidates)))
                    for idx, br in enumerate(branch_candidates):
                        col = cols[idx % len(cols)]
                        with col:
                            snippet = br.trace.get_full_text()[-160:]
                            st.markdown(f"**{br.label}**")
                            st.caption(snippet)
                            if st.button(f"Adopt {br.label}", key=f"adopt_{selected_idx}_{idx}"):
                                st.session_state['trace'] = br.trace
                                st.session_state['analysis'] = br.analysis
                                st.session_state['selected_token_index'] = st.session_state.get('branch_origin_idx', selected_idx)
                                st.session_state['rethink_start_index'] = (st.session_state.get('branch_origin_idx', selected_idx) or 0) + 1
                                st.session_state['hidden_state_cache'] = {}
                                st.session_state['probe_cache'] = {}
                                st.session_state.pop('analysis_obj', None)
                                st.session_state['branch_candidates'] = []
                                st.session_state['branch_origin_idx'] = None
                                st.rerun()

                    if st.button("Discard branches"):
                        st.session_state['branch_candidates'] = []
                        st.session_state['branch_origin_idx'] = None
                else:
                    st.info("No branch candidates yet.")

            with tabs[3]:
                st.subheader("Hidden States & Explanations")
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

                        header_col, control_col = st.columns([3, 2], gap="small")
                        with header_col:
                            st.markdown("Select a layer to inspect:")

                        with control_col:
                            min_cols = st.selectbox(
                                "Min buttons per row",
                                options=list(range(2, 33)),
                                index=list(range(2, 33)).index(8),
                                key=f"layer_min_cols_{selected_idx}",
                                help="Lower bound for how many layer buttons appear per row",
                                label_visibility="collapsed",
                            )
                            st.caption("Min buttons per row")

                        total_layers = len(layer_ids)

                        def _calc_cols(n: int) -> int:
                            # More layers → more columns (wide screens pack more), capped for readability
                            return max(min_cols, min(3, math.floor(n / 4) + min_cols))

                        grid_size = _calc_cols(total_layers)

                        for row_start in range(0, total_layers, grid_size):
                            cols = st.columns(min(grid_size, total_layers - row_start), gap="small")
                            for offset, layer_id in enumerate(layer_ids[row_start:row_start + grid_size]):
                                with cols[offset]:
                                    if st.button(f"L{layer_id}", key=f"layer_{selected_idx}_{layer_id}"):
                                        st.session_state['selected_layer'] = layer_id
                                        st.rerun()

                        layer_state = trajectory.get_by_layer(st.session_state['selected_layer'])
                        if layer_state:
                            logit_lens = session.controller._decode_hidden_state(layer_state.get_value(), top_k=10)
                            lens_df = pd.DataFrame(logit_lens, columns=["Token", "Prob"])

                            lens_col, explain_col = st.columns(2)
                            with lens_col:
                                st.markdown(f"**Logit Lens @ Layer {st.session_state['selected_layer']}:**")
                                st.dataframe(lens_df.style.format({"Prob": "{:.4f}"}), hide_index=True, width='stretch')

                            probe_key = f"{selected_idx}_{st.session_state['selected_layer']}"
                            if probe_key not in probe_cache:
                                with st.spinner("Running explanation probe..."):
                                    probe_cache[probe_key] = session.controller.probe_state(
                                        trace,
                                        selected_idx,
                                        layer_idx=st.session_state['selected_layer'],
                                        cached_state=layer_state,
                                        cached_trajectory=trajectory,
                                    )

                            probe_result = probe_cache.get(probe_key, {})
                            with explain_col:
                                st.markdown("**State Explanation:**")
                                st.write(probe_result.get("explanation", ""))
                        else:
                            st.info("Layer state unavailable.")
                    else:
                        st.info("No hidden states captured for this token.")
                else:
                    st.info("Hidden state not available for this token.")

        else:
            st.info("Select a token.")

else:
    st.info("Please load a model to begin.")
