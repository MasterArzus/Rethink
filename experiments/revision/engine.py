"""Shared runtime for staged revision experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from checker import CheckResult, RevisionChecker, first_repair_position
from prompt import (
    MAX_TURNS,
    build_cd_proxy_prompt,
    build_chat_actor_prompt,
    build_local_repair_prompt,
    build_reflexion_prompt,
    build_steer_actor_prompt,
    build_task_prompt,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "staged_cases.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

MODELS = {
    "deepseek_r1": "/root/autodl-fs/deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llama3_8b": "/root/autodl-fs/LLM-Research/Meta-Llama-3.1-8B-Instruct",
    "qwen3_8b": "/root/autodl-fs/Qwen/Qwen3-8B",
    "deepseek_r1_qwen_1_5b": "/root/autodl-fs/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "qwen2_5_1_5b": "/root/autodl-fs/Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2_5_14b_instruct": "/root/autodl-fs/Qwen/Qwen2.5-14B-Instruct",
}


@dataclass
class StageRecord:
    case_id: str
    case_type: str
    method: str
    model: str
    k: int
    passed: bool
    checker_message: str
    failed_stage: Optional[int]
    failure_type: str
    prompt_text: str
    answer: str
    model_time_seconds: float
    inspect_time_seconds: float
    generated_tokens: int
    prompt_tokens: int
    total_tokens: int
    actor_time_seconds: float = 0.0
    actor_tokens: int = 0
    actor_instruction: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    case_id: str
    case_type: str
    method: str
    model: str
    success: bool
    pass_k: int
    final_answer: str
    total_model_time_seconds: float
    total_inspect_time_seconds: float
    total_generated_tokens: int
    total_prompt_tokens: int
    total_tokens: int
    total_actor_tokens: int
    stage_records: List[StageRecord]


def parse_common_args(method: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run staged revision experiment: {method}")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", action="append", dest="models", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--use-llm-judge", action="store_true")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--actor-model", default=os.environ.get("ACTOR_MODEL", "MiniMax-M2.7"))
    parser.add_argument("--actor-api-base", default=os.environ.get("MINIMAX_API_BASE", "https://api.minimaxi.com/anthropic"))
    parser.add_argument("--actor-api-key", default=os.environ.get("MINIMAX_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument("--resume", action="store_true", help="Resume from existing CSV output and skip completed case IDs")
    parser.add_argument("--do-sample", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true", help="Run prompt/checker plumbing without loading local models")
    parser.add_argument("--method", default=method)
    return parser.parse_args()


def load_cases(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = payload.get("cases", payload if isinstance(payload, list) else [])
    return cases[:limit] if limit else cases


def resolve_models(selected: Optional[List[str]]) -> Dict[str, str]:
    if not selected:
        return {"qwen2_5_1_5b": MODELS["qwen2_5_1_5b"]}
    resolved = {}
    for name in selected:
        resolved[name] = MODELS.get(name, name)
    return resolved


def apply_chat_template(tokenizer, messages: List[Dict[str, str]]):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    return tokenizer(text, return_tensors="pt").input_ids


def generate_response(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int, int, float]:
    start = time.perf_counter()
    inputs = apply_chat_template(tokenizer, messages).to(model.device)
    prompt_tokens = int(inputs.shape[1])
    kwargs = {"max_new_tokens": max_new_tokens}
    if tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    if generation_kwargs:
        kwargs.update(generation_kwargs)
    with torch.no_grad():
        outputs = model.generate(inputs, **kwargs)
    generated_ids = outputs[0][prompt_tokens:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, prompt_tokens, int(len(generated_ids)), time.perf_counter() - start


def dry_generate(prompt: str, case: Dict[str, Any], k: int) -> Tuple[str, int, int, float]:
    text = f"Please consider {case.get('topic', 'the topic')} with a polite note about {case.get('topic', 'it')}."
    if case["type"] == "json" and k >= 2:
        keys = []
        for stage in case.get("stages", []):
            if stage.get("type") == "json":
                keys = stage.get("constraints", {}).get("keys", [])
                break
        text = json.dumps({key: f"please {key} for {case.get('topic', 'topic')}" for key in keys}, ensure_ascii=False)
    return text, len(prompt.split()), len(text.split()), 0.001


def human_inspect_time(answer: str, seconds_per_word: float = 0.1, thinking_seconds: float = 5.0) -> float:
    return len(answer.split()) * seconds_per_word + thinking_seconds


def make_stage_record(
    case: Dict[str, Any],
    method: str,
    model_name: str,
    k: int,
    answer: str,
    check: CheckResult,
    model_time: float,
    inspect_time: float,
    prompt_tokens: int,
    generated_tokens: int,
    actor_tokens: int = 0,
    prompt_text: str = "",
    actor_instruction: str = "",
    actor_time_seconds: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageRecord:
    return StageRecord(
        case_id=case["id"],
        case_type=case["type"],
        method=method,
        model=model_name,
        k=k,
        passed=check.passed,
        checker_message=check.message,
        failed_stage=check.failed_stage,
        failure_type=check.failure_type,
        prompt_text=prompt_text,
        answer=answer,
        model_time_seconds=model_time,
        inspect_time_seconds=inspect_time,
        actor_time_seconds=actor_time_seconds,
        generated_tokens=generated_tokens,
        prompt_tokens=prompt_tokens,
        total_tokens=prompt_tokens + generated_tokens,
        actor_tokens=actor_tokens,
        actor_instruction=actor_instruction,
        metadata=metadata or {},
    )


def summarize_case(case: Dict[str, Any], method: str, model_name: str, records: List[StageRecord]) -> CaseResult:
    final = records[-1] if records else None
    pass_records = [r for r in records if r.passed and (r.k >= 4 or r.k == MAX_TURNS)]
    pass_k = pass_records[0].k if pass_records else 0
    success = bool(pass_records)
    return CaseResult(
        case_id=case["id"],
        case_type=case["type"],
        method=method,
        model=model_name,
        success=success,
        pass_k=pass_k,
        final_answer=final.answer if final else "",
        total_model_time_seconds=sum(r.model_time_seconds for r in records),
        total_inspect_time_seconds=sum(r.inspect_time_seconds for r in records),
        total_generated_tokens=sum(r.generated_tokens for r in records),
        total_prompt_tokens=sum(r.prompt_tokens for r in records),
        total_tokens=sum(r.total_tokens for r in records),
        total_actor_tokens=sum(r.actor_tokens for r in records),
        stage_records=records,
    )


class ActorClient:
    def __init__(self, model: str, api_base: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model
        self.api_base = api_base
        self.client = None
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("MINIMAX_API_KEY")
        if api_key:
            import anthropic

            self.client = anthropic.Anthropic(api_key=api_key, base_url=api_base) if api_base else anthropic.Anthropic(api_key=api_key)

    def call(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> Tuple[str, int, float]:
        if self.client is None:
            return "", 0, 0.0
        start = time.perf_counter()
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
        usage = getattr(msg, "usage", None)
        tokens = 0
        if usage:
            tokens = int(getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0))
        return text.strip(), tokens, time.perf_counter() - start


def run_reflexion_case(ctx, case: Dict[str, Any]) -> CaseResult:
    records: List[StageRecord] = []
    answer = ""
    for k in range(1, ctx.max_turns + 1):
        if k <= 4:
            prompt = build_task_prompt(case, k, answer if answer else None)
        else:
            prev = records[-1].checker_message if records else ""
            prompt = build_reflexion_prompt(case, 4, answer, prev)
        answer, prompt_tokens, gen_tokens, model_time = ctx.generate(prompt)
        check = ctx.checker.check(case, answer, min(k, 4))
        records.append(make_stage_record(case, ctx.method, ctx.model_name, k, answer, check, model_time, 0.0, prompt_tokens, gen_tokens, prompt_text=prompt))
        if k >= 4 and check.passed:
            break
    return summarize_case(case, ctx.method, ctx.model_name, records)


def run_auto_lr_case(ctx, case: Dict[str, Any]) -> CaseResult:
    records: List[StageRecord] = []
    answer = ""
    for k in range(1, ctx.max_turns + 1):
        upto = min(k, 4)
        if k <= 4 or not records:
            prompt = build_task_prompt(case, upto, answer if answer else None)
        else:
            pos = first_repair_position(answer, records[-1])
            prefix = answer[:pos]
            prompt = build_local_repair_prompt(case, 4, prefix, records[-1].checker_message)
        new_answer, prompt_tokens, gen_tokens, model_time = ctx.generate(prompt)
        if k > 4 and answer:
            pos = first_repair_position(answer, records[-1])
            answer = answer[:pos] + new_answer
        else:
            answer = new_answer
        check = ctx.checker.check(case, answer, upto)
        records.append(make_stage_record(case, ctx.method, ctx.model_name, k, answer, check, model_time, 0.0, prompt_tokens, gen_tokens, prompt_text=prompt))
        if k >= 4 and check.passed:
            break
    return summarize_case(case, ctx.method, ctx.model_name, records)


def cd_generation_kwargs(case: Dict[str, Any], k: int, tokenizer) -> Dict[str, Any]:
    if k not in (1, 2):
        return {}
    forbidden: List[str] = []
    for stage in case.get("stages", []):
        if int(stage.get("k", 0)) <= k and stage.get("type") == "taboo":
            forbidden.extend(stage.get("constraints", {}).get("forbidden_words", []))
    if not forbidden:
        return {}
    bad_words_ids = []
    seen = set()
    for word in forbidden:
        for variant in {word, word.lower(), word.capitalize(), f" {word}", f" {word.lower()}"}:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            key = tuple(ids)
            if ids and key not in seen:
                bad_words_ids.append(ids)
                seen.add(key)
    return {"bad_words_ids": bad_words_ids} if bad_words_ids else {}


def run_cd_case(ctx, case: Dict[str, Any]) -> CaseResult:
    records: List[StageRecord] = []
    answer = ""
    for k in range(1, ctx.max_turns + 1):
        upto = min(k, 4)
        prompt = build_cd_proxy_prompt(case, upto)
        kwargs = cd_generation_kwargs(case, upto, ctx.tokenizer) if ctx.tokenizer else {}
        answer, prompt_tokens, gen_tokens, model_time = ctx.generate(prompt, kwargs)
        check = ctx.checker.check(case, answer, upto)
        meta = {"cd_mode": "native" if upto <= 2 else "proxy" if upto == 3 else "not_applicable_dynamic"}
        records.append(make_stage_record(case, ctx.method, ctx.model_name, k, answer, check, model_time, 0.0, prompt_tokens, gen_tokens, prompt_text=prompt, metadata=meta))
        if k >= 4:
            break
    return summarize_case(case, ctx.method, ctx.model_name, records)


def run_actor_case(ctx, case: Dict[str, Any], mode: str, lite: bool = False) -> CaseResult:
    records: List[StageRecord] = []
    answer = ""
    messages: List[Dict[str, str]] = []
    for k in range(1, ctx.max_turns + 1):
        upto = min(k, 4)
        instruction = ""
        actor_tokens = 0
        actor_time = 0.0
        actor_prompt = ""
        if k <= 4 or not records:
            user_prompt = build_task_prompt(case, upto, answer if answer else None)
        else:
            prev = records[-1]
            if mode == "chat":
                actor_prompt = build_chat_actor_prompt(case, 4, answer, prev.checker_message)
            else:
                actor_prompt = build_steer_actor_prompt(case, 4, answer, prev.checker_message, lite=lite)
            instruction, actor_tokens, actor_time = ctx.actor.call(actor_prompt)
            if mode == "chat":
                user_prompt = f"{build_task_prompt(case, 4, answer)}\n\nHuman feedback: {instruction or prev.checker_message}"
            else:
                pos = first_repair_position(answer, prev)
                prefix = answer[:pos]
                user_prompt = build_local_repair_prompt(case, 4, prefix, instruction or prev.checker_message)
        if mode == "chat":
            messages.append({"role": "user", "content": user_prompt})
            answer, prompt_tokens, gen_tokens, model_time = ctx.generate_messages(messages)
            messages.append({"role": "assistant", "content": answer})
        else:
            answer, prompt_tokens, gen_tokens, model_time = ctx.generate(user_prompt)
        check = ctx.checker.check(case, answer, upto)
        stage_inspect = human_inspect_time(answer) + actor_time
        records.append(
            make_stage_record(
                case,
                ctx.method,
                ctx.model_name,
                k,
                answer,
                check,
                model_time,
                stage_inspect,
                prompt_tokens,
                gen_tokens,
                actor_tokens,
                prompt_text=user_prompt,
                actor_instruction=instruction,
                actor_time_seconds=actor_time,
                metadata={"actor_prompt": actor_prompt} if actor_prompt else {},
            )
        )
        if k >= 4 and check.passed:
            break
    return summarize_case(case, ctx.method, ctx.model_name, records)


@dataclass
class RuntimeContext:
    method: str
    model_name: str
    model: Any
    tokenizer: Any
    checker: RevisionChecker
    actor: ActorClient
    max_turns: int
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    dry_run: bool = False

    def generation_kwargs(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"do_sample": self.do_sample}
        if self.do_sample:
            kwargs.update({"temperature": self.temperature, "top_p": self.top_p})
        if extra:
            kwargs.update(extra)
        return kwargs

    def generate(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[str, int, int, float]:
        if self.dry_run:
            return dry_generate(prompt, {"id": "dry", "type": "taboo", "topic": "dry run"}, 1)
        return generate_response(
            self.model,
            self.tokenizer,
            [{"role": "user", "content": prompt}],
            self.max_new_tokens,
            self.generation_kwargs(generation_kwargs),
        )

    def generate_messages(self, messages: List[Dict[str, str]]) -> Tuple[str, int, int, float]:
        if self.dry_run:
            prompt = "\n".join(m["content"] for m in messages)
            return prompt[-200:], len(prompt.split()), 20, 0.001
        return generate_response(self.model, self.tokenizer, messages, self.max_new_tokens, self.generation_kwargs())


def load_model(model_name: str, model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def write_outputs(results: List[CaseResult], output_dir: Path, method: str, model_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"{model_name}_{method}"
    flat: List[Dict[str, Any]] = []
    for result in results:
        for record in result.stage_records:
            row = asdict(record)
            row["prompt_text"] = row["prompt_text"].replace("\r", "\\r").replace("\n", "\\n")
            row["answer"] = row["answer"].replace("\r", "\\r").replace("\n", "\\n")
            row["checker_message"] = row["checker_message"].replace("\r", "\\r").replace("\n", "\\n")
            row["actor_instruction"] = row["actor_instruction"].replace("\r", "\\r").replace("\n", "\\n")
            row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False)
            flat.append(row)
    if flat:
        tmp_path = base.with_suffix(".csv.tmp")
        final_path = base.with_suffix(".csv")
        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()))
            writer.writeheader()
            writer.writerows(flat)
        tmp_path.replace(final_path)


def _bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _optional_int(value: Any) -> Optional[int]:
    if value in ("", None):
        return None
    return int(value)


def _stage_from_csv(row: Dict[str, str]) -> StageRecord:
    metadata_text = row.get("metadata") or "{}"
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError:
        metadata = {"raw": metadata_text}
    return StageRecord(
        case_id=row["case_id"],
        case_type=row["case_type"],
        method=row["method"],
        model=row["model"],
        k=int(row["k"]),
        passed=_bool_value(row["passed"]),
        checker_message=row.get("checker_message", ""),
        failed_stage=_optional_int(row.get("failed_stage")),
        failure_type=row.get("failure_type", ""),
        prompt_text=row.get("prompt_text", ""),
        answer=row.get("answer", ""),
        model_time_seconds=float(row.get("model_time_seconds") or 0.0),
        inspect_time_seconds=float(row.get("inspect_time_seconds") or 0.0),
        actor_time_seconds=float(row.get("actor_time_seconds") or 0.0),
        generated_tokens=int(float(row.get("generated_tokens") or 0)),
        prompt_tokens=int(float(row.get("prompt_tokens") or 0)),
        total_tokens=int(float(row.get("total_tokens") or 0)),
        actor_tokens=int(float(row.get("actor_tokens") or 0)),
        actor_instruction=row.get("actor_instruction", ""),
        metadata=metadata,
    )


def _case_from_records(records: List[StageRecord]) -> CaseResult:
    records = sorted(records, key=lambda item: item.k)
    first = records[0]
    final = records[-1]
    pass_records = [r for r in records if r.passed and (r.k >= 4 or r.k == MAX_TURNS)]
    pass_k = pass_records[0].k if pass_records else 0
    return CaseResult(
        case_id=first.case_id,
        case_type=first.case_type,
        method=first.method,
        model=first.model,
        success=bool(pass_records),
        pass_k=pass_k,
        final_answer=final.answer,
        total_model_time_seconds=sum(r.model_time_seconds for r in records),
        total_inspect_time_seconds=sum(r.inspect_time_seconds for r in records),
        total_generated_tokens=sum(r.generated_tokens for r in records),
        total_prompt_tokens=sum(r.prompt_tokens for r in records),
        total_tokens=sum(r.total_tokens for r in records),
        total_actor_tokens=sum(r.actor_tokens for r in records),
        stage_records=records,
    )


def load_existing_results(output_dir: Path, method: str, model_name: str) -> List[CaseResult]:
    path = output_dir / f"{model_name}_{method}.csv"
    if not path.exists():
        return []
    grouped: Dict[str, List[StageRecord]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["case_id"], []).append(_stage_from_csv(row))
    return [_case_from_records(records) for records in grouped.values() if records]


def resolve_sampling(args: argparse.Namespace) -> bool:
    if args.do_sample == "true":
        return True
    if args.do_sample == "false":
        return False
    return args.method in {"chat", "steer", "steer_lite"}


def run_method(args: argparse.Namespace, runner) -> None:
    cases = load_cases(args.dataset_path, args.limit)
    model_map = resolve_models(args.models)
    output_dir = Path(args.output_dir)
    checker = RevisionChecker(use_llm_judge=args.use_llm_judge, judge_model=args.judge_model)
    actor = ActorClient(args.actor_model, args.actor_api_base, args.actor_api_key)
    if args.method in {"chat", "steer", "steer_lite"} and actor.client is None:
        raise RuntimeError(f"{args.method} requires MINIMAX_API_KEY, ANTHROPIC_API_KEY, or --actor-api-key.")

    for model_name, model_path in model_map.items():
        model = tokenizer = None
        if not args.dry_run:
            print(f"Loading {model_name}: {model_path}", flush=True)
            model, tokenizer = load_model(model_name, model_path)
        do_sample = resolve_sampling(args)
        ctx = RuntimeContext(
            method=args.method,
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            checker=checker,
            actor=actor,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            dry_run=args.dry_run,
        )
        results = load_existing_results(output_dir, args.method, model_name) if args.resume else []
        completed = {result.case_id for result in results}
        if completed:
            print(f"[{model_name}/{args.method}] resume: loaded {len(completed)} completed cases", flush=True)
        for i, case in enumerate(cases, 1):
            if case["id"] in completed:
                print(f"[{model_name}/{args.method}] skip {i}/{len(cases)} {case['id']} (completed)", flush=True)
                continue
            print(f"[{model_name}/{args.method}] {i}/{len(cases)} {case['id']}", flush=True)
            results.append(runner(ctx, case))
            write_outputs(results, output_dir, args.method, model_name)
        if model is not None:
            del model
            del tokenizer
            torch.cuda.empty_cache()
