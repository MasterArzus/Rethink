import streamlit as st
import pandas as pd
import os
import math
import glob
import re
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
from rethink.server.experiment_logger import ExperimentLogger
from rethink.server.component import render_token_stream
from rethink.utils.config import GenerationConfig, PromptConfig, LoggingConfig
from st_click_detector import click_detector
from dataset.ifeval.checkers import get_checker

st.set_page_config(layout="wide", page_title="Rethink")

st.title("LLM Interactive Framework")

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


def strip_think_content(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    trailing_open = cleaned.find("<think>")
    if trailing_open != -1:
        cleaned = cleaned[:trailing_open]
    return cleaned


def model_config_sort_key(filename):
    name = filename.lower()
    if "1_5b" in name or "1.5b" in name:
        return (0, name)
    if "8b" in name:
        return (1, name)
    if "13b" in name or "14b" in name:
        return (2, name)
    return (3, name)

def get_condition_label():
    return "chat" if st.session_state.get("experiment_mode") == "Baseline (Chat)" else "steer"

def get_experiment_logger():
    return st.session_state.get("experiment_logger")

def build_task_key(task_context, model_name):
    return json.dumps(
        {
            "task_id": task_context.get("task_id"),
            "task_type": task_context.get("task_type"),
            "dataset_name": task_context.get("dataset_name"),
            "condition": get_condition_label(),
            "model_name": model_name,
        },
        sort_keys=True,
    )

def set_task_context(task_id, task_type, dataset_name, prompt_text, metadata=None):
    st.session_state["current_task_context"] = {
        "task_id": str(task_id),
        "task_type": task_type or "unknown",
        "dataset_name": dataset_name,
        "prompt_text": prompt_text,
        "metadata": metadata or {},
    }
    st.session_state["suppress_task_autostart"] = None

def finish_active_task(session, success, final_checker_message=None, failure_reason=None, metadata=None):
    logger = get_experiment_logger()
    if not logger or not logger.active_task:
        return None

    finished_key = st.session_state.get("active_logged_task_key")
    summary = session.finish_task(
        success=success,
        final_checker_message=final_checker_message,
        failure_reason=failure_reason,
        metadata=metadata,
    )
    st.session_state["active_logged_task_key"] = None
    st.session_state["last_logged_checker_trace_id"] = None
    if finished_key:
        st.session_state["suppress_task_autostart"] = finished_key
    return summary

def ensure_task_started(session, model_name):
    logger = get_experiment_logger()
    task_context = st.session_state.get("current_task_context")
    if not logger or not task_context:
        return

    task_key = build_task_key(task_context, model_name)
    if st.session_state.get("active_logged_task_key") == task_key:
        return
    if st.session_state.get("suppress_task_autostart") == task_key:
        return

    if logger.active_task:
        finish_active_task(session, success=False, failure_reason="task_switched")

    session.start_task(
        task_id=task_context["task_id"],
        task_type=task_context["task_type"],
        condition=get_condition_label(),
        model_name=model_name,
        dataset_name=task_context["dataset_name"],
        metadata=task_context.get("metadata", {}),
    )
    st.session_state["active_logged_task_key"] = task_key
    st.session_state["last_logged_checker_trace_id"] = None

def ensure_ad_hoc_task(session, model_name, prompt_text):
    logger = get_experiment_logger()
    if logger and logger.active_task:
        return

    set_task_context(
        task_id=f"adhoc-{uuid.uuid4()}",
        task_type="freeform",
        dataset_name="freeform",
        prompt_text=prompt_text,
        metadata={"source": "manual_chat_input"},
    )
    ensure_task_started(session, model_name)

def estimate_token_count(tokenizer, text):
    if not tokenizer or not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return 0

# =============================================================================
# SIDEBAR: Reorganized Configuration
# =============================================================================

# --- Session State Defaults (persist selections) ---
if 'default_model_config' not in st.session_state:
    st.session_state['default_model_config'] = "llama3_8b.yaml"
if 'default_dataset' not in st.session_state:
    st.session_state['default_dataset'] = "IFEval (Steerability)"

# Initialize metrics in session state if not present
if 'total_tokens_used' not in st.session_state:
    st.session_state['total_tokens_used'] = 0
if 'interaction_turns' not in st.session_state:
    st.session_state['interaction_turns'] = 0
if 'correction_tokens' not in st.session_state:
    st.session_state['correction_tokens'] = 0
if 'correction_turns' not in st.session_state:
    st.session_state['correction_turns'] = 0

# ============================================================================
# Section 1: Experiment Setup (Collapsible)
# ============================================================================
with st.sidebar.expander("🧪 实验设置", expanded=True):
    # Participant ID
    participant_id = st.text_input("Participant ID", value=st.session_state.get("participant_id", "pilot"))
    st.session_state["participant_id"] = participant_id

    # Experiment Mode
    experiment_mode = st.radio(
        "Mode",
        ["Baseline (Chat)", "Rethink (Steering)"],
        key="experiment_mode",
        horizontal=True
    )

    # Model Selection
    model_configs = load_config_files("models")

    # Filter to only show models with valid paths
    valid_model_configs = {}
    for cfg_name, cfg_path in model_configs.items():
        cfg_data = load_yaml(cfg_path) or {}
        model_path = str(cfg_data.get("model_name_or_path", "") or "").strip()
        if model_path and os.path.exists(model_path):
            valid_model_configs[cfg_name] = cfg_path

    model_config_options = sorted(valid_model_configs.keys(), key=model_config_sort_key)

    # Ensure default is still valid
    if st.session_state['default_model_config'] not in model_config_options:
        st.session_state['default_model_config'] = model_config_options[0] if model_config_options else None

    default_idx = model_config_options.index(st.session_state['default_model_config']) if st.session_state['default_model_config'] in model_config_options else 0
    selected_model_config_file = st.selectbox(
        "Model",
        options=model_config_options,
        index=default_idx
    )
    # Remember selection
    st.session_state['default_model_config'] = selected_model_config_file

    # Load model config
    model_cfg_path = model_configs[selected_model_config_file]
    model_cfg_data = load_yaml(model_cfg_path) or {}
    model_cfg_data.setdefault("device_map", "auto")
    model_cfg_data.setdefault("torch_dtype", "float16")
    model_cfg_data.setdefault("attn_implementation", "eager")
    model_path = str(model_cfg_data.get("model_name_or_path", "") or "").strip()

    # Model path validation
    if model_path and os.path.exists(model_path):
        st.caption("✓ Model path valid")
    elif model_path:
        st.caption("⚠ Model path not found")

    # Logger info
    logger = get_experiment_logger()
    if logger:
        logger.update_participant(participant_id)
        with st.expander("Logger Info"):
            st.caption(f"Session: {logger.session_id[:8]}...")
            st.caption(f"Events: {logger.events_path.name}")

# ============================================================================
# Section 2: Task Selection (Collapsible)
# ============================================================================
with st.sidebar.expander("📋 任务选择", expanded=True):
    dataset_config = load_yaml("/root/Rethink/configs/datasets.yaml")
    dataset_options = ["IFEval (Steerability)", "Hydrology QA"] + list(dataset_config.keys())
    default_dataset_idx = dataset_options.index(st.session_state['default_dataset']) if st.session_state['default_dataset'] in dataset_options else 0
    selected_dataset_name = st.selectbox("Dataset", options=dataset_options, index=default_dataset_idx)
    st.session_state['default_dataset'] = selected_dataset_name

    loaded_example_question = None

    if selected_dataset_name == "IFEval (Steerability)":
        ifeval_path = "/root/Rethink/dataset/ifeval/taskset_60_hard.json"
        if os.path.exists(ifeval_path):
            with open(ifeval_path, 'r') as f:
                data_obj = json.load(f)
                ifeval_tasks = data_obj.get("tasks", [])

            task_type_display = st.selectbox("Task Type", ["All", "Part 1: Taboo Hard (30)", "Part 2: JSON Hard (30)"])

            if task_type_display == "Part 1: Taboo Hard (30)":
                filtered_data = [t for t in ifeval_tasks if t['type'] == "taboo"]
            elif task_type_display == "Part 2: JSON Hard (30)":
                filtered_data = [t for t in ifeval_tasks if t['type'] == "json"]
            else:
                filtered_data = ifeval_tasks

            total_count = len(filtered_data)
            max_idx = max(0, total_count - 1)
            example_idx = st.number_input(f"Task Index (Total: {total_count})", min_value=0, max_value=max_idx, value=0)

            if filtered_data:
                selected_task = filtered_data[example_idx]
                question = selected_task.get("prompt", "")
                constraints = selected_task.get("constraints", {})
                task_type = selected_task.get("type", "")

                current_selection_state = (task_type_display, example_idx)
                if 'last_ifeval_selection' not in st.session_state:
                    st.session_state['last_ifeval_selection'] = None

                if st.session_state['last_ifeval_selection'] != current_selection_state:
                    st.session_state.messages = []
                    st.session_state.pop('trace', None)
                    st.session_state.pop('analysis', None)

                    st.session_state['user_input_area'] = question
                    st.session_state['current_constraints'] = constraints
                    st.session_state['current_task_type'] = task_type
                    st.session_state['last_ifeval_selection'] = current_selection_state
                    set_task_context(
                        task_id=selected_task.get("id", f"ifeval-{task_type}-{example_idx}"),
                        task_type=task_type,
                        dataset_name="ifeval",
                        prompt_text=question,
                        metadata={
                            "filter": task_type_display,
                            "example_index": int(example_idx),
                        },
                    )

                    st.session_state['total_tokens_used'] = 0
                    st.session_state['interaction_turns'] = 0
                    st.session_state['correction_tokens'] = 0
                    st.session_state['correction_turns'] = 0

                    st.rerun()

                with st.expander("Preview Task"):
                    st.markdown(f"**Type:** {task_type}")
                    st.markdown(f"**Prompt:** {question[:200]}..." if len(question) > 200 else f"**Prompt:** {question}")
                    st.json(constraints)
        else:
            st.error("IFEval task set not found")

    elif selected_dataset_name == "Hydrology QA":
        hydro_base_path = "/root/Rethink/dataset/hydrology_qa"
        if os.path.exists(hydro_base_path):
            json_files = glob.glob(os.path.join(hydro_base_path, "*.json"))
            json_options = [os.path.basename(f) for f in json_files]
            if json_options:
                selected_file = st.selectbox("JSON File", json_options)
                full_path = os.path.join(hydro_base_path, selected_file)

                with open(full_path, 'r', encoding='utf-8') as f:
                    hydro_data = json.load(f)

                total_count = len(hydro_data)
                max_idx = max(0, total_count - 1)
                example_idx = st.number_input(f"Item Index (Total: {total_count})", min_value=0, max_value=max_idx, value=0)

                if hydro_data:
                    selected_item = hydro_data[example_idx]
                    instruction = selected_item.get("instruction", "")
                    inp = selected_item.get("input", "")
                    output = selected_item.get("output", "")

                    question = f"{instruction}\n\n{inp}" if inp else instruction

                    current_hydro_selection = (selected_file, example_idx)
                    if 'last_hydro_selection' not in st.session_state:
                        st.session_state['last_hydro_selection'] = None

                    if st.session_state['last_hydro_selection'] != current_hydro_selection:
                        st.session_state.messages = []
                        st.session_state.pop('trace', None)
                        st.session_state.pop('analysis', None)
                        st.session_state['user_input_area'] = question

                        st.session_state['total_tokens_used'] = 0
                        st.session_state['interaction_turns'] = 0
                        st.session_state['correction_tokens'] = 0
                        st.session_state['correction_turns'] = 0

                        st.session_state.pop('last_ifeval_selection', None)
                        st.session_state.pop('current_constraints', None)
                        st.session_state.pop('current_task_type', None)
                        set_task_context(
                            task_id=f"hydrology-{selected_file}-{example_idx}",
                            task_type="hydrology_qa",
                            dataset_name="hydrology_qa",
                            prompt_text=question,
                            metadata={
                                "file": selected_file,
                                "example_index": int(example_idx),
                            },
                        )

                        st.session_state['last_hydro_selection'] = current_hydro_selection
                        st.rerun()

                with st.expander("Preview"):
                    st.markdown(f"**Instruction:** {instruction[:100]}..." if len(instruction) > 100 else f"**Instruction:** {instruction}")
                    if inp:
                        st.markdown(f"**Input:** {inp[:100]}..." if len(inp) > 100 else f"**Input:** {inp}")
            else:
                st.warning("No JSON files found")
        else:
            st.error("Hydrology QA path not found")

    elif selected_dataset_name != "None":
        dataset_path = dataset_config[selected_dataset_name]
        try:
            if os.path.exists(dataset_path):
                if os.path.isdir(dataset_path):
                    ds = datasets.load_from_disk(dataset_path)
                else:
                    st.warning("Dataset path is not a directory")
                    ds = datasets.load_dataset(dataset_path)

                if isinstance(ds, datasets.DatasetDict):
                    split = st.selectbox("Split", options=list(ds.keys()))
                    data = ds[split]
                else:
                    data = ds

                max_idx = len(data) - 1
                example_idx = st.number_input("Example Index", min_value=0, max_value=max_idx, value=0)

                selected_example = data[example_idx]
                question = selected_example.get("question", "")
                answer = selected_example.get("answer", "")

                with st.expander("Preview"):
                    st.markdown(f"**Q:** {question[:150]}..." if len(question) > 150 else f"**Q:** {question}")

                if st.button("Load Example"):
                    st.session_state['user_input_area'] = question
                    set_task_context(
                        task_id=f"{selected_dataset_name}-{example_idx}",
                        task_type=selected_dataset_name,
                        dataset_name=selected_dataset_name,
                        prompt_text=question,
                        metadata={"example_index": int(example_idx)},
                    )

                    st.session_state['total_tokens_used'] = 0
                    st.session_state['interaction_turns'] = 0
                    st.session_state['correction_tokens'] = 0
                    st.session_state['correction_turns'] = 0

                    st.rerun()
            else:
                st.error("Dataset path not found")

        except Exception as e:
            st.error(f"Error loading dataset: {e}")

# ============================================================================
# Section 3: Generation Parameters (Collapsible)
# ============================================================================
with st.sidebar.expander("⚙️ 生成参数", expanded=False):
    gen_configs = load_config_files("generation")
    selected_gen_config_file = st.selectbox(
        "Preset",
        options=list(gen_configs.keys()) + ["Default"]
    )

    if selected_gen_config_file != "Default":
        gen_cfg_path = gen_configs[selected_gen_config_file]
        gen_cfg_data = load_yaml(gen_cfg_path)
    else:
        gen_cfg_data = GenerationConfig().to_dict()

    # Core parameters
    temperature = st.slider("Temperature", 0.0, 2.0, float(gen_cfg_data.get("temperature", 0.4)), 0.1)
    top_p = st.slider("Top P", 0.0, 1.0, float(gen_cfg_data.get("top_p", 0.7)), 0.05)
    max_new_tokens = st.slider("Max Tokens", 16, 2048, int(gen_cfg_data.get("max_new_tokens", 512)), 16)

    gen_cfg_data.update({
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    })

    with st.expander("Advanced"):
        repetition_penalty = st.slider(
            "Repetition Penalty", 1.0, 2.0,
            float(gen_cfg_data.get("repetition_penalty", 1.0)), 0.05
        )
        no_repeat_ngram_size = st.number_input(
            "No Repeat N-Gram", 0, 10,
            int(gen_cfg_data.get("no_repeat_ngram_size", 0))
        )
        hide_think_for_display = st.checkbox(
            "Hide <think>", value=True
        )
        gen_cfg_data.update({
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size
        })

    # Prompt Config (load from file or use default)
    prompt_configs = load_config_files("prompts")
    prompt_options = list(prompt_configs.keys()) + ["Default"]
    # Default to ifeval.yaml if available
    default_prompt_idx = prompt_options.index("ifeval.yaml") if "ifeval.yaml" in prompt_options else len(prompt_options) - 1

    selected_prompt_config_file = st.selectbox(
        "Prompt Preset",
        options=prompt_options,
        index=default_prompt_idx
    )
    if selected_prompt_config_file != "Default":
        prompt_cfg_path = prompt_configs[selected_prompt_config_file]
        prompt_cfg_data = load_yaml(prompt_cfg_path)
    else:
        prompt_cfg_data = PromptConfig().to_dict()

# ============================================================================
# Section 4: Logging Control (Manual Start)
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📋 实验记录")

# Session state for logging control
if 'logging_started' not in st.session_state:
    st.session_state['logging_started'] = False
if 'current_participant_id' not in st.session_state:
    st.session_state['current_participant_id'] = st.session_state.get("participant_id", "pilot")

# Display logging status
if st.session_state['logging_started']:
    st.sidebar.success("✅ 记录中")
    logger = get_experiment_logger()
    if logger:
        st.sidebar.caption(f"Participant: {logger.participant_id}")
        st.sidebar.caption(f"Session: {logger.session_id[:8]}...")

    # Show live metrics from session
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Tokens", st.session_state.get('total_tokens_used', 0))
        st.metric("Clicks", st.session_state.get('total_clicks', 0))
    with col2:
        st.metric("Gen Calls", st.session_state.get('interaction_turns', 0))
        st.metric("Probes", st.session_state.get('total_probes', 0))

    if st.sidebar.button("🔄 重置记录"):
        # Reset all metrics
        st.session_state['total_tokens_used'] = 0
        st.session_state['interaction_turns'] = 0
        st.session_state['correction_tokens'] = 0
        st.session_state['correction_turns'] = 0
        st.session_state['total_clicks'] = 0
        st.session_state['total_probes'] = 0
        st.session_state['total_branch_actions'] = 0
        st.rerun()

    if st.sidebar.button("⏹️ 停止记录"):
        st.session_state['logging_started'] = False
        st.rerun()
else:
    st.sidebar.info("点击开始记录实验")
    if st.sidebar.button("▶️ 开始记录"):
        # Initialize/reset logger with current participant ID
        participant = st.session_state.get("participant_id", "pilot")
        logging_cfg = LoggingConfig(output_dir="/root/Rethink/outputs/interactive_sessions")
        experiment_logger = ExperimentLogger(logging_cfg, participant_id=participant)
        st.session_state['experiment_logger'] = experiment_logger
        st.session_state['logging_started'] = True

        # Reset metrics
        st.session_state['total_tokens_used'] = 0
        st.session_state['interaction_turns'] = 0
        st.session_state['correction_tokens'] = 0
        st.session_state['correction_turns'] = 0
        st.session_state['total_clicks'] = 0
        st.session_state['total_probes'] = 0
        st.session_state['total_branch_actions'] = 0

        st.rerun()

st.sidebar.caption("记录在 outputs/interactive_sessions/")

# ============================================================================
# Model Loading
# ============================================================================
model_cfg_errors = [] if not model_path else []
if st.sidebar.button("🚀 Load Model", disabled=bool(model_cfg_errors)):
    try:
        with st.status("Loading model resources...", expanded=True) as status:
            st.write("Initializing Session Manager...")

            st.write(f"Loading model weights from {model_path}...")
            model, tokenizer = SessionManager.get_resources(model_path, model_cfg_data)

            st.write("Initializing interactive session...")
            # Use existing experiment_logger if logging has been started, otherwise None
            experiment_logger = st.session_state.get('experiment_logger')

            st.session_state['model'] = model
            st.session_state['tokenizer'] = tokenizer
            st.session_state['interactive_session'] = InteractiveSession(model, tokenizer, experiment_logger=experiment_logger)
            # Reset model-coupled runtime state after switching model to avoid stale traces.
            st.session_state['messages'] = []
            st.session_state.pop('trace', None)
            st.session_state.pop('analysis', None)
            st.session_state.pop('analysis_obj', None)
            st.session_state['hidden_state_cache'] = {}
            st.session_state['probe_cache'] = {}
            st.session_state['branch_candidates'] = []
            st.session_state['branch_origin_idx'] = None
            st.session_state['selected_layer'] = None
            st.session_state['selected_token_index'] = 0
            st.session_state['rethink_start_index'] = -1
            st.session_state['trace_id'] = str(uuid.uuid4())
            st.session_state['active_logged_task_key'] = None
            st.session_state['last_logged_checker_trace_id'] = None
            st.session_state['suppress_task_autostart'] = None
            
            status.update(label="Model loaded successfully!", state="complete", expanded=False)
            
        st.success("Ready to generate!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# --- Main Interface ---
if 'model' in st.session_state:
    session = st.session_state['interactive_session']
    if get_experiment_logger() and session.experiment_logger is None:
        session.set_experiment_logger(get_experiment_logger())
    ensure_task_started(session, model_path)

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
                        
                        checker_text = strip_think_content(full_text).strip() or full_text
                    
                        try:
                            checker = get_checker(task_type)
                            passed, error_msg = checker.check(checker_text, constraints)
                            current_trace_id = st.session_state.get('trace_id')
                            if st.session_state.get("last_logged_checker_trace_id") != current_trace_id:
                                session.record_checker_result(
                                    passed,
                                    error_msg,
                                    trace_id=current_trace_id,
                                    metadata={"task_type": task_type},
                                )
                                st.session_state["last_logged_checker_trace_id"] = current_trace_id
                                if passed:
                                    finish_active_task(
                                        session,
                                        success=True,
                                        final_checker_message=error_msg,
                                        metadata={"task_type": task_type},
                                    )
                        
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
                                    session.log_event(
                                        "token_selected",
                                        token_index=new_idx,
                                        selected_token=token_data[new_idx]['token'],
                                        trace_id=st.session_state.get('trace_id'),
                                    )
                                    st.rerun()
                            except:
                                pass
                    
                        # =================================================================
                        # NEW: Two-Column Token Analysis Panel
                        # =================================================================
                        st.divider()
                        if 0 <= selected_idx < len(token_data):
                            # Initialize components if needed
                            if 'analysis_obj' not in st.session_state:
                                from rethink.analysis.trace_analysis import TraceAnalysis
                                st.session_state['analysis_obj'] = TraceAnalysis(trace, session.controller.model, session.controller.tokenizer)

                            analyzer = st.session_state['analysis_obj']

                            hs_cache = st.session_state.setdefault('hidden_state_cache', {})
                            probe_cache = st.session_state.setdefault('probe_cache', {})
                            st.session_state.setdefault('branch_candidates', [])
                            st.session_state.setdefault('branch_origin_idx', None)

                            # ---- Two-Column Layout ----
                            col_left, col_right = st.columns([1, 1], gap="medium")

                            # === LEFT COLUMN: Layer Analysis ===
                            with col_left:
                                st.markdown("### 🔬 层分析")

                                # Ensure Hidden States are computed
                                if selected_idx not in hs_cache:
                                    with st.spinner("Computing hidden states..."):
                                        hs_cache[selected_idx] = session.controller.compute_hidden_states_for_step(trace, selected_idx, layers=None)

                                token_state = hs_cache.get(selected_idx)
                                if token_state:
                                    trajectory = token_state.trajectory
                                    layer_ids = sorted(trajectory.to_dict().keys())
                                    total_layers = len(layer_ids)

                                    if layer_ids:
                                        selected_layer = st.session_state.get('selected_layer')
                                        if selected_layer is None or selected_layer not in layer_ids:
                                            selected_layer = layer_ids[-1]
                                            st.session_state['selected_layer'] = selected_layer

                                        # Layer slider
                                        layer_idx = layer_ids.index(selected_layer) if selected_layer in layer_ids else len(layer_ids) - 1
                                        selected_layer = st.slider(
                                            "Layer",
                                            0, total_layers - 1,
                                            layer_idx,
                                            key=f"layer_slider_{selected_idx}",
                                            format=f"L{layer_ids[min(layer_idx, total_layers-1)] if layer_idx < total_layers else layer_idx}"
                                        )
                                        actual_layer = layer_ids[min(selected_layer, total_layers - 1)] if total_layers > 0 else 0

                                        if actual_layer != st.session_state.get('selected_layer'):
                                            st.session_state['selected_layer'] = actual_layer
                                            session.log_event("layer_changed", token_index=selected_idx, trace_id=st.session_state.get('trace_id'), metadata={"layer": actual_layer})

                                        # Logit Lens for selected layer (as clickable tokens)
                                        layer_state = trajectory.get_by_layer(st.session_state['selected_layer'])
                                        if layer_state:
                                            logit_lens = session.controller._decode_hidden_state(layer_state.get_value(), top_k=10)

                                            st.markdown(f"**Logit Lens @ Layer {st.session_state['selected_layer']}**")

                                            # Show as compact dataframe
                                            lens_display = [{"Token": tok, "Prob": f"{prob:.4f}"} for tok, prob in logit_lens[:10]]
                                            st.dataframe(pd.DataFrame(lens_display), hide_index=True, use_container_width=True)

                                            # Store tok_options for use in right column
                                            st.session_state[f'lens_toks_{selected_idx}'] = [tok for tok, _ in logit_lens[:10]]
                                else:
                                    st.info("Hidden state not available")

                            # === RIGHT COLUMN: Quick Actions ===
                            with col_right:
                                st.markdown("### 🔧 快速操作")

                                # Steering prompt
                                steering_prompt = st.text_input(
                                    "Steering prompt (可选)",
                                    key=f"steer_{selected_idx}",
                                    help="在重新生成时添加额外的引导提示"
                                )

                                # Selection for Force token
                                tok_options = st.session_state.get(f'lens_toks_{selected_idx}', [])
                                current_selection = st.session_state.get('selected_alt_token')
                                if current_selection not in tok_options:
                                    current_selection = tok_options[0] if tok_options else None
                                if tok_options:
                                    selected_tok = st.selectbox(
                                        "Selection",
                                        tok_options,
                                        index=tok_options.index(current_selection) if current_selection in tok_options else 0
                                    )
                                    if selected_tok:
                                        st.session_state['selected_alt_token'] = selected_tok

                                # Action buttons
                                col_act1, col_act2 = st.columns(2)
                                with col_act1:
                                    if st.button("🔄 Truncate & Retry", key=f"truncate_{selected_idx}", use_container_width=True):
                                        session.log_event("truncate_clicked", token_index=selected_idx, trace_id=st.session_state.get('trace_id'))
                                        with st.spinner(f"Rethinking from step {selected_idx}..."):
                                            session.start_generation()
                                            new_trace, new_analysis = session.rethink_from_step(
                                                trace, selected_idx,
                                                max_new_tokens=gen_cfg_data.get("max_new_tokens", 128),
                                                steering_prompt=steering_prompt,
                                            )
                                            tokens_gen = len(new_trace.tokenlist) - selected_idx
                                            st.session_state['total_tokens_used'] += tokens_gen
                                            st.session_state['interaction_turns'] += 1
                                            st.session_state['correction_tokens'] += tokens_gen
                                            st.session_state['correction_turns'] += 1
                                            st.session_state['trace_id'] = str(uuid.uuid4())
                                            session.record_generation(tokens_gen, is_correction=True, trace_id=st.session_state['trace_id'], metadata={"operation": "truncate_retry", "token_index": selected_idx})

                                            st.session_state['trace'] = new_trace
                                            st.session_state['analysis'] = new_analysis
                                            st.session_state['selected_token_index'] = selected_idx
                                            st.session_state['rethink_start_index'] = selected_idx + 1
                                            st.session_state.messages[-1]["content"] = new_trace.get_full_text()
                                            st.rerun()

                                with col_act2:
                                    selected_alt_token = st.session_state.get('selected_alt_token')
                                    if selected_alt_token:
                                        if st.button(f"⚡ Force `{selected_alt_token}`", key=f"force_{selected_idx}", use_container_width=True):
                                            session.log_event("force_retry_clicked", token_index=selected_idx, selected_token=selected_alt_token, trace_id=st.session_state.get('trace_id'))
                                            with st.spinner(f"Forcing '{selected_alt_token}'..."):
                                                session.start_generation()
                                                new_trace, new_analysis = session.rethink_from_step(
                                                    trace, selected_idx,
                                                    max_new_tokens=gen_cfg_data.get("max_new_tokens", 128),
                                                    force_token=selected_alt_token,
                                                    steering_prompt=steering_prompt,
                                                )
                                                tokens_gen = len(new_trace.tokenlist) - selected_idx
                                                st.session_state['total_tokens_used'] += tokens_gen
                                                st.session_state['interaction_turns'] += 1
                                                st.session_state['correction_tokens'] += tokens_gen
                                                st.session_state['correction_turns'] += 1
                                                st.session_state['trace_id'] = str(uuid.uuid4())
                                                session.record_generation(tokens_gen, is_correction=True, trace_id=st.session_state['trace_id'], metadata={"operation": "force_retry", "token_index": selected_idx, "forced_token": selected_alt_token})

                                                st.session_state['trace'] = new_trace
                                                st.session_state['analysis'] = new_analysis
                                                st.session_state['selected_token_index'] = selected_idx
                                                st.session_state['rethink_start_index'] = selected_idx
                                                st.session_state.messages[-1]["content"] = new_trace.get_full_text()
                                                st.rerun()
                                    else:
                                        st.button("⚡ Force & Retry", key=f"force_disabled_{selected_idx}", disabled=True, use_container_width=True)

                                st.markdown("---")

                                # Branching section (collapsible)
                                with st.expander("🌿 Branching", expanded=False):
                                    branch_k = st.slider("K branches", 2, 5, 3)
                                    branch_strategy = st.radio("Strategy", ["sample", "beam"], horizontal=True, key=f"branch_strat_{selected_idx}")
                                    branch_steer = st.text_input("Branch prompt (optional)", key=f"branch_steer_{selected_idx}")

                                    if st.button("Generate Branches", key=f"gen_branch_{selected_idx}"):
                                        session.log_event("branch_generate_clicked", token_index=selected_idx, trace_id=st.session_state.get('trace_id'), metadata={"k": branch_k, "strategy": branch_strategy})
                                        with st.spinner("Generating branches..."):
                                            branches = session.branch_from_step(trace, selected_idx, k=branch_k, strategy=branch_strategy, max_new_tokens=gen_cfg_data.get("max_new_tokens", 128), steering_prompt=branch_steer)
                                            tokens_generated = sum(len(b.trace.tokenlist) - selected_idx for b in branches)
                                            st.session_state['total_tokens_used'] += tokens_generated
                                            st.session_state['interaction_turns'] += 1
                                            st.session_state['correction_tokens'] += tokens_generated
                                            st.session_state['correction_turns'] += 1
                                            session.record_generation(tokens_generated, is_correction=True, trace_id=st.session_state.get('trace_id'), metadata={"operation": "branch_generate", "token_index": selected_idx, "branch_count": len(branches)})
                                            st.session_state['branch_candidates'] = branches
                                            st.session_state['branch_origin_idx'] = selected_idx
                                            st.rerun()

                                    # Show branch candidates
                                    branch_candidates = st.session_state.get('branch_candidates', [])
                                    if branch_candidates:
                                        st.markdown("**Candidates:**")
                                        for idx, br in enumerate(branch_candidates):
                                            snippet = br.trace.get_full_text()[-120:]
                                            with st.container():
                                                col_b, col_a = st.columns([4, 1])
                                                with col_b:
                                                    st.caption(f"{br.label}: ...{snippet}")
                                                with col_a:
                                                    if st.button("Use", key=f"adopt_{selected_idx}_{idx}"):
                                                        session.log_event("branch_adopted", token_index=st.session_state.get('branch_origin_idx', selected_idx), trace_id=st.session_state.get('trace_id'), metadata={"branch_label": br.label, "branch_index": idx})
                                                        st.session_state['trace'] = br.trace
                                                        st.session_state['analysis'] = br.analysis
                                                        st.session_state['selected_token_index'] = st.session_state.get('branch_origin_idx', selected_idx)
                                                        st.session_state['rethink_start_index'] = (st.session_state.get('branch_origin_idx', selected_idx) or 0) + 1
                                                        st.session_state['hidden_state_cache'] = {}
                                                        st.session_state['probe_cache'] = {}
                                                        st.session_state.pop('analysis_obj', None)
                                                        st.session_state['branch_candidates'] = []
                                                        st.session_state['branch_origin_idx'] = None
                                                        st.session_state['trace_id'] = str(uuid.uuid4())
                                                        st.session_state.messages[-1]["content"] = br.trace.get_full_text()
                                                        st.rerun()
                                        if st.button("Discard All", key=f"discard_{selected_idx}"):
                                            st.session_state['branch_candidates'] = []
                                            st.session_state['branch_origin_idx'] = None
                                            st.rerun()

                                # Analysis Layer Meaning (below branching)
                                probe_key = f"{selected_idx}_{st.session_state.get('selected_layer', 0)}"
                                if st.button("🔍 Analyze Layer Meaning", key=f"analyze_{probe_key}", use_container_width=True):
                                    session.log_event("probe_requested", token_index=selected_idx, trace_id=st.session_state.get('trace_id'), metadata={"layer": st.session_state.get('selected_layer', 0)})
                                    with st.spinner("Analyzing..."):
                                        probe_cache[probe_key] = session.controller.probe_state(trace, selected_idx, layer_idx=st.session_state.get('selected_layer', 0), cached_state=layer_state, cached_trajectory=trajectory)
                                    st.rerun()

                                # Show probe result
                                probe_result = probe_cache.get(probe_key, {})
                                explanation_raw = probe_result.get("explanation")
                                if explanation_raw:
                                    import re
                                    try:
                                        match = re.search(r'```json\s*(.*?)\s*```', explanation_raw, re.DOTALL)
                                        if match:
                                            parsed_json = json.loads(match.group(1))
                                        elif explanation_raw.strip().startswith('{'):
                                            parsed_json = json.loads(explanation_raw)
                                        else:
                                            parsed_json = None

                                        if parsed_json and isinstance(parsed_json, dict):
                                            preds = parsed_json.get("token_predictions")
                                            if isinstance(preds, list):
                                                for p in preds[:3]:
                                                    with st.expander(f"{p.get('token', '')} ({p.get('confidence', 0):.2%})"):
                                                        st.write(p.get('reasoning', ''))
                                            else:
                                                st.json(parsed_json)
                                        else:
                                            st.info(explanation_raw[:500])
                                    except:
                                        st.info(explanation_raw[:500] if explanation_raw else "No analysis available")


    # Chat Input
    col_input, col_run, col_clear = st.columns([8, 1, 1])

    with col_run:
        if st.button("Rerun", help="Regenerate response and reset metrics"):
            if st.session_state.messages:
                # If last message is assistant, remove it to regenerate
                if st.session_state.messages[-1]["role"] == "assistant":
                    st.session_state.messages.pop()
                
                # Reset metrics for this new run
                st.session_state['total_tokens_used'] = 0
                st.session_state['interaction_turns'] = 0
                st.session_state['suppress_task_autostart'] = None
                
                st.session_state['trigger_generation'] = True
                st.rerun()

    with col_clear:
        if st.button("Clear", help="Start a new conversation"):
            finish_active_task(session, success=False, failure_reason="cleared")
            st.session_state.messages = []
            st.session_state.pop('trace', None)
            st.session_state.pop('analysis', None)
            st.session_state.pop('current_task_context', None)
            st.rerun()

    prompt = st.chat_input("Type your message...")

    # Check if we need to trigger generation (either from chat input or sidebar load)
    if prompt or st.session_state.get('trigger_generation', False):
        ensure_task_started(session, model_path)
        if not get_experiment_logger() or not get_experiment_logger().active_task:
            fallback_prompt = prompt or (st.session_state.messages[-1]["content"] if st.session_state.messages else "")
            ensure_ad_hoc_task(session, model_path, fallback_prompt)

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            session.log_event(
                "user_text_submitted",
                metadata={
                    "char_count": len(prompt),
                    "token_count": estimate_token_count(session.controller.tokenizer, prompt),
                },
            )
    
        # Reset trigger
        st.session_state['trigger_generation'] = False
    
        session.log_event(
            "generate_clicked",
            metadata={
                "source": "chat_input" if prompt else "rerun_or_loaded_task",
                "message_count": len(st.session_state.messages),
            },
        )
    
        with st.chat_message("assistant"):
            live_stream_placeholder = st.empty()
            stream_tokens = []

            def token_stream_callback(token):
                if token:
                    token = token.replace("\uFFFD", "")
                stream_tokens.append(token)
                streamed = "".join(stream_tokens)
                if hide_think_for_display:
                    streamed = strip_think_content(streamed)
                live_stream_placeholder.markdown(streamed + "▌")

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

                session.start_generation()
                trace, analysis = session.run_initial_inference(
                    full_prompt_str,
                    use_template=False, # We already applied the template
                    max_new_tokens=max_new_tokens,
                    stream_callback=token_stream_callback,
                )

                # Update metrics
                tokens_gen = len(trace.tokenlist)
                st.session_state['total_tokens_used'] += tokens_gen
                st.session_state['interaction_turns'] += 1
                st.session_state['trace_id'] = str(uuid.uuid4())
                session.record_generation(
                    tokens_gen,
                    is_correction=len(st.session_state.messages) > 1,
                    trace_id=st.session_state['trace_id'],
                    metadata={
                        "source": "chat_input" if prompt else "rerun_or_loaded_task",
                    },
                )
                
                # If this is a correction (history > 1), update correction metrics
                if len(st.session_state.messages) > 1:
                    st.session_state['correction_tokens'] += tokens_gen
                    st.session_state['correction_turns'] += 1

                full_response = "".join(stream_tokens)
                
                display_response = full_response.replace("\uFFFD", "")
                if hide_think_for_display:
                    display_response = strip_think_content(display_response).strip()
                
                live_stream_placeholder.markdown(display_response)

                st.session_state['trace'] = trace
                st.session_state['analysis'] = analysis
                st.session_state['rethink_start_index'] = -1
                st.session_state['hidden_state_cache'] = {}
                st.session_state['probe_cache'] = {}
                st.session_state['selected_layer'] = None
                st.session_state.pop('analysis_obj', None)
                st.session_state['selected_token_index'] = 0
            
                st.session_state.messages.append({"role": "assistant", "content": display_response})
                st.rerun()

else:
    st.info("Please load a model to begin.")
