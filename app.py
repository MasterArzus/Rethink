import streamlit as st
import pandas as pd
import os
import math
import glob
import yaml
import sys
import datasets
import uuid
import json
from datetime import datetime

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rethink.server.session_manager import SessionManager
from rethink.server.interactive import InteractiveSession
from rethink.server.component import render_token_stream
from rethink.utils.config import GenerationConfig, PromptConfig
from st_click_detector import click_detector
from dataset.ifeval.checkers import get_checker

st.set_page_config(layout="wide", page_title="Rethink")

st.title("LLM Interactive Framework")

# --- Logging Setup ---
if 'action_log' not in st.session_state:
    st.session_state['action_log'] = []

def log_action(action_type, details):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action_type,
        "details": details,
        "mode": st.session_state.get('experiment_mode', 'unknown')
    }
    st.session_state['action_log'].append(entry)

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

# Experiment Mode
experiment_mode = st.sidebar.radio(
    "Experiment Mode",
    ["Baseline (Chat)", "Rethink (Steering)"],
    key="experiment_mode"
)

# Download Logs
if st.sidebar.button("Download Logs"):
    logs = st.session_state['action_log']
    json_logs = json.dumps(logs, indent=2)
    st.sidebar.download_button(
        label="Save Log File",
        data=json_logs,
        file_name=f"experiment_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# --- Metrics Display for Volunteers ---
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Session Metrics")
st.sidebar.info("Please record these values in your form after finishing the task.")

# Initialize metrics in session state if not present
if 'total_tokens_used' not in st.session_state:
    st.session_state['total_tokens_used'] = 0
if 'interaction_turns' not in st.session_state:
    st.session_state['interaction_turns'] = 0

col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Total Tokens", st.session_state['total_tokens_used'])
with col2:
    st.metric("Turns", st.session_state['interaction_turns'])

st.sidebar.markdown("---")

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
dataset_options = ["IFEval (Steerability)"] + list(dataset_config.keys())
selected_dataset_name = st.sidebar.selectbox("Select Dataset", options=dataset_options)

loaded_example_question = None

if selected_dataset_name == "IFEval (Steerability)":
    ifeval_path = "/root/Rethink/dataset/ifeval/taskset_120.json"
    if os.path.exists(ifeval_path):
        with open(ifeval_path, 'r') as f:
            data_obj = json.load(f)
            ifeval_tasks = data_obj.get("tasks", [])
        
        # Filter by type if needed, or just show all
        task_type_filter = st.sidebar.selectbox("Filter Task Type", ["All", "forbidden_words", "json_format"])
        
        if task_type_filter != "All":
            filtered_data = [t for t in ifeval_tasks if t['type'] == task_type_filter]
        else:
            filtered_data = ifeval_tasks
            
        total_count = len(filtered_data)
        max_idx = max(0, total_count - 1)
        example_idx = st.sidebar.number_input(f"Task Index (Total: {total_count})", min_value=0, max_value=max_idx, value=0)
        
        if filtered_data:
            selected_task = filtered_data[example_idx]
            question = selected_task.get("prompt", "")
            constraints = selected_task.get("constraints", {})
            task_type = selected_task.get("type", "")
            
            # Auto-load when selection changes
            current_selection_state = (task_type_filter, example_idx)
            if 'last_ifeval_selection' not in st.session_state:
                st.session_state['last_ifeval_selection'] = None
                
            if st.session_state['last_ifeval_selection'] != current_selection_state:
                # Clear previous session
                st.session_state.messages = []
                st.session_state.pop('trace', None)
                st.session_state.pop('analysis', None)
                
                st.session_state['user_input_area'] = question
                st.session_state['current_constraints'] = constraints
                st.session_state['current_task_type'] = task_type
                st.session_state['last_ifeval_selection'] = current_selection_state
                
                # Reset metrics on task change
                st.session_state['total_tokens_used'] = 0
                st.session_state['interaction_turns'] = 0
                
                st.rerun()
            
            with st.sidebar.expander("Preview Task"):
                st.markdown(f"**Type:** {task_type}")
                st.markdown(f"**Prompt:** {question}")
                st.json(constraints)
    else:
        st.sidebar.error(f"IFEval task set not found at {ifeval_path}")

elif selected_dataset_name != "None":
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
                
                # Reset metrics on example load
                st.session_state['total_tokens_used'] = 0
                st.session_state['interaction_turns'] = 0
                
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

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Apply Prompt Template Logic
    system_prompt = prompt_cfg_data.get("system_prompt", "")
    user_role = prompt_cfg_data.get("user_role", "user")

    # Handle "Load Task" or "Load Example" from sidebar
    # If user_input_area was set by sidebar, we treat it as a pending message to send
    if 'user_input_area' in st.session_state and st.session_state['user_input_area']:
        pending_prompt = st.session_state.pop('user_input_area')
        # Only add if it's different from the last user message to avoid duplicates on rerun
        # or just force add it.
        st.session_state.messages.append({"role": "user", "content": pending_prompt})
        # Trigger generation immediately
        st.session_state['trigger_generation'] = True

    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
            # If this is the LAST message and it is from assistant, render the interactive trace if available
            if i == len(st.session_state.messages) - 1 and message["role"] == "assistant":
                if 'trace' in st.session_state:
                    trace = st.session_state['trace']
                    analysis = st.session_state['analysis']
                
                    # --- Checker Integration (Only for the latest trace) ---
                    if 'current_constraints' in st.session_state and 'current_task_type' in st.session_state:
                        constraints = st.session_state['current_constraints']
                        task_type = st.session_state['current_task_type']
                        full_text = trace.get_full_text()
                    
                        try:
                            checker = get_checker(task_type)
                            passed, error_msg = checker.check(full_text, constraints)
                        
                            if passed:
                                st.success("✅ Constraint Satisfied!")
                            else:
                                st.error(f"❌ Constraint Violated: {error_msg}")
                        except Exception as e:
                            st.warning(f"Could not run checker: {e}")

                    if experiment_mode == "Rethink (Steering)":
                        st.divider()
                        st.caption("Interactive Steering Mode")
                    
                        selected_idx = st.session_state.get('selected_token_index', 0)
                        token_data = []
                        rethink_start = st.session_state.get('rethink_start_index', -1)

                        for idx, token in enumerate(trace.tokenlist):
                            is_critical = False
                            reason = ""
                            for interval in analysis:
                                if interval.start <= idx <= interval.end:
                                    is_critical = True
                                    reason = interval.type
                                    break

                            is_new = (rethink_start != -1 and idx >= rethink_start)

                            token_data.append({
                                "index": idx, "token": token.token, "prob": token.prob,
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
                                    log_action("select_token", {"index": new_idx, "token": token_data[new_idx]['token']})
                                    st.rerun()
                            except:
                                pass
                    
                        # Render Analysis Tabs below the token stream
                        st.divider()
                        if 0 <= selected_idx < len(token_data):
                            selected_item = token_data[selected_idx]

                            if 'analysis_obj' not in st.session_state:
                                from rethink.analysis.trace_analysis import TraceAnalysis
                                st.session_state['analysis_obj'] = TraceAnalysis(trace, session.controller.model, session.controller.tokenizer)

                            analyzer = st.session_state['analysis_obj']
                        
                            # We need to cache alternatives to avoid re-running on every interaction
                            if f"alts_{selected_idx}" not in st.session_state:
                                with st.spinner("Decoding alternatives..."):
                                    st.session_state[f"alts_{selected_idx}"] = analyzer.get_token_alternatives(selected_idx, k=10)
                        
                            alts = st.session_state[f"alts_{selected_idx}"]

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
                                                log_action("truncate", {"step": selected_idx})
                                                with st.spinner(f"Rethinking from step {selected_idx}..."):
                                                    new_trace, new_analysis = session.rethink_from_step(
                                                        trace,
                                                        selected_idx,
                                                        max_new_tokens=gen_cfg_data.get("max_new_tokens", 128),
                                                        steering_prompt=steering_prompt,
                                                    )
                                                    # Update metrics
                                                    st.session_state['total_tokens_used'] += (len(new_trace.tokenlist) - selected_idx)
                                                    st.session_state['interaction_turns'] += 1
                                                
                                                    st.session_state['trace'] = new_trace
                                                    st.session_state['analysis'] = new_analysis
                                                    st.session_state['selected_token_index'] = selected_idx
                                                    st.session_state['rethink_start_index'] = selected_idx + 1
                                                    # Update the last message content
                                                    st.session_state.messages[-1]["content"] = new_trace.get_full_text()
                                                    st.rerun()

                                        with col_btn2:
                                            selected_alt_token = st.session_state.get('selected_alt_token')
                                            if selected_alt_token:
                                                if st.button("Rethink with Selection", help="Replace current token with selected alternative and regenerate."):
                                                    log_action("rethink_force", {"step": selected_idx, "token": selected_alt_token})
                                                    with st.spinner(f"Forcing '{selected_alt_token}' and rethinking..."):
                                                        new_trace, new_analysis = session.rethink_from_step(
                                                            trace,
                                                            selected_idx,
                                                            max_new_tokens=gen_cfg_data.get("max_new_tokens", 128),
                                                            force_token=selected_alt_token,
                                                            steering_prompt=steering_prompt,
                                                        )
                                                        # Update metrics
                                                        st.session_state['total_tokens_used'] += (len(new_trace.tokenlist) - selected_idx)
                                                        st.session_state['interaction_turns'] += 1

                                                        st.session_state['trace'] = new_trace
                                                        st.session_state['analysis'] = new_analysis
                                                        st.session_state['selected_token_index'] = selected_idx
                                                        st.session_state['rethink_start_index'] = selected_idx
                                                        # Update the last message content
                                                        st.session_state.messages[-1]["content"] = new_trace.get_full_text()
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
                                        # Update metrics for branches
                                        tokens_generated = sum(len(b.trace.tokenlist) - selected_idx for b in branches)
                                        st.session_state['total_tokens_used'] += tokens_generated
                                        st.session_state['interaction_turns'] += 1

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
                                                # Update message content
                                                st.session_state.messages[-1]["content"] = br.trace.get_full_text()
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
                                        st.info("No layers available in trajectory.")
                                else:
                                    st.info("Hidden state not available for this token.")

    # Chat Input
    col_input, col_run, col_clear = st.columns([8, 1, 1])

    with col_run:
        if st.button("Run", help="Regenerate response"):
            if st.session_state.messages:
                # If last message is assistant, remove it to regenerate
                if st.session_state.messages[-1]["role"] == "assistant":
                    st.session_state.messages.pop()
                st.session_state['trigger_generation'] = True
                st.rerun()

    with col_clear:
        if st.button("Clear", help="Start a new conversation"):
            st.session_state.messages = []
            st.session_state.pop('trace', None)
            st.session_state.pop('analysis', None)
            st.rerun()

    prompt = st.chat_input("Type your message...")

    # Check if we need to trigger generation (either from chat input or sidebar load)
    if prompt or st.session_state.get('trigger_generation', False):
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
    
        # Reset trigger
        st.session_state['trigger_generation'] = False
    
        log_action("generate", {"prompt": prompt})
    
        with st.chat_message("assistant"):
            live_stream_placeholder = st.empty()
            stream_tokens = []

            def token_stream_callback(token):
                stream_tokens.append(token)
                live_stream_placeholder.markdown("".join(stream_tokens) + "▌")

            with st.spinner("Generating trace..."):
                session.controller.cfg.prompt = PromptConfig(**prompt_cfg_data)
                session.controller.cfg.generation = GenerationConfig(**gen_cfg_data)

                # Construct full conversation prompt
                full_prompt_str = ""
                if hasattr(session.controller.tokenizer, "apply_chat_template") and session.controller.tokenizer.chat_template:
                    # Build messages list including system prompt
                    messages = [{"role": prompt_cfg_data.get('system_role', 'system'), "content": system_prompt}]
                    # Append history
                    for msg in st.session_state.messages:
                        # Map 'user'/'assistant' to configured roles if needed, but usually standard roles work
                        role = prompt_cfg_data.get('user_role', 'user') if msg['role'] == 'user' else 'assistant'
                        messages.append({"role": role, "content": msg['content']})
                
                    full_prompt_str = session.controller.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    # Fallback manual construction
                    full_prompt_str = f"{system_prompt}\n\n"
                    for msg in st.session_state.messages:
                        role = "Question" if msg['role'] == 'user' else "Answer"
                        full_prompt_str += f"{role}: {msg['content']}\n"
                    full_prompt_str += "Answer:"

                trace, analysis = session.run_initial_inference(
                    full_prompt_str,
                    use_template=False, # We already applied the template
                    max_new_tokens=max_new_tokens,
                    stream_callback=token_stream_callback,
                )

                # Update metrics
                st.session_state['total_tokens_used'] += len(trace.tokenlist)
                st.session_state['interaction_turns'] += 1

                full_response = "".join(stream_tokens)
                live_stream_placeholder.markdown(full_response)

                st.session_state['trace'] = trace
                st.session_state['analysis'] = analysis
                st.session_state['rethink_start_index'] = -1
                st.session_state['hidden_state_cache'] = {}
                st.session_state['probe_cache'] = {}
                st.session_state['selected_layer'] = None
                st.session_state.pop('analysis_obj', None)
                st.session_state['selected_token_index'] = 0
                st.session_state['trace_id'] = str(uuid.uuid4())
            
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.rerun()

else:
    st.info("Please load a model to begin.")
