import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from rethink.utils.config import LoggingConfig


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskState:
    task_id: str
    task_type: str
    dataset_name: str
    condition: str
    model_name: str
    participant_id: str
    session_id: str
    trace_id: Optional[str] = None
    task_start_timestamp: str = field(default_factory=_utc_now_iso)
    last_event_timestamp: str = field(default_factory=_utc_now_iso)
    task_end_timestamp: Optional[str] = None
    generation_started_at: Optional[str] = None
    generation_time_seconds: float = 0.0
    success: Optional[bool] = None
    failure_reason: Optional[str] = None
    final_checker_message: Optional[str] = None
    checker_fail_count: int = 0
    number_of_generate_calls: int = 0
    number_of_correction_generate_calls: int = 0
    total_generated_tokens: int = 0
    correction_tokens: int = 0
    final_turn_index: int = 0
    first_pass_turn: Optional[int] = None
    total_action_count: int = 0
    total_clicks: int = 0
    total_probes: int = 0
    total_branch_actions: int = 0
    total_user_messages: int = 0
    total_user_characters: int = 0
    total_user_prompt_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExperimentLogger:
    """Persist interactive experiment events and task summaries to disk."""

    def __init__(
        self,
        logging_config: Optional[LoggingConfig] = None,
        participant_id: str = "anonymous",
        session_id: Optional[str] = None,
    ):
        self.config = logging_config or LoggingConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.output_dir / self.config.events_file
        self.task_summary_path = self.output_dir / self.config.task_summary_file
        self.task_summary_csv_path = self.output_dir / self.config.task_summary_csv

        self.participant_id = participant_id or "anonymous"
        self.session_id = session_id or str(uuid4())
        self.active_task: Optional[TaskState] = None

    def update_participant(self, participant_id: Optional[str]) -> None:
        if participant_id:
            self.participant_id = participant_id

    def _append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _append_task_summary_csv(self, summary: Dict[str, Any]) -> None:
        write_header = not self.task_summary_csv_path.exists()
        with self.task_summary_csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(summary)

    def start_task(
        self,
        task_id: str,
        task_type: str,
        condition: str,
        model_name: str,
        dataset_name: str = "unknown",
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskState:
        if self.active_task and self.active_task.task_id != task_id:
            self.finish_task(success=False, failure_reason="task_switched")

        self.active_task = TaskState(
            task_id=task_id,
            task_type=task_type,
            dataset_name=dataset_name,
            condition=condition,
            model_name=model_name,
            participant_id=self.participant_id,
            session_id=self.session_id,
            trace_id=trace_id,
            metadata=metadata or {},
        )
        self.log_event("task_loaded", metadata=metadata)
        self.log_event("task_started", metadata=metadata)
        return self.active_task

    def log_event(
        self,
        event_type: str,
        *,
        mode: Optional[str] = None,
        token_index: Optional[int] = None,
        selected_token: Optional[str] = None,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.active_task:
            return None

        timestamp = _utc_now_iso()
        task = self.active_task
        previous_ts = datetime.fromisoformat(task.last_event_timestamp)
        current_ts = datetime.fromisoformat(timestamp)
        task.last_event_timestamp = timestamp
        if trace_id is not None:
            task.trace_id = trace_id
        task.total_action_count += 1

        if event_type in {"token_selected", "truncate_clicked", "force_retry_clicked",
                           "sos_highlight_opened", "logit_lens_opened", "explanation_opened"}:
            task.total_clicks += 1
        if event_type == "probe_requested":
            task.total_probes += 1
        if event_type in {"branch_generate_clicked", "branch_adopted"}:
            task.total_branch_actions += 1
        if event_type == "user_text_submitted":
            task.total_user_messages += 1
            task.total_user_characters += int((metadata or {}).get("char_count", 0))
            task.total_user_prompt_tokens += int((metadata or {}).get("token_count", 0))

        record = {
            "timestamp": timestamp,
            "participant_id": task.participant_id,
            "session_id": task.session_id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "dataset_name": task.dataset_name,
            "trace_id": task.trace_id,
            "mode": mode or task.condition,
            "event_type": event_type,
            "token_index": token_index,
            "selected_token": selected_token,
            "latency_since_task_start": self._seconds_since(task.task_start_timestamp, timestamp),
            "latency_since_previous_event": (current_ts - previous_ts).total_seconds(),
            "metadata": metadata or {},
        }
        self._append_jsonl(self.events_path, record)
        return record

    def start_generation(self) -> None:
        if not self.active_task:
            return
        self.active_task.generation_started_at = _utc_now_iso()
        self.log_event("generation_started")

    def record_generation(
        self,
        generated_tokens: int,
        *,
        is_correction: bool,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.active_task:
            return

        task = self.active_task
        task.number_of_generate_calls += 1
        task.final_turn_index = task.number_of_generate_calls
        task.total_generated_tokens += int(generated_tokens)
        if is_correction:
            task.number_of_correction_generate_calls += 1
            task.correction_tokens += int(generated_tokens)

        duration = 0.0
        if task.generation_started_at:
            duration = self._seconds_since(task.generation_started_at, _utc_now_iso())
            task.generation_time_seconds += duration
            task.generation_started_at = None

        record_metadata = {
            "generated_tokens": int(generated_tokens),
            "is_correction": bool(is_correction),
            "generation_duration_seconds": duration,
        }
        if metadata:
            record_metadata.update(metadata)
        self.log_event("generation_completed", trace_id=trace_id, metadata=record_metadata)

    def record_checker_result(
        self,
        passed: bool,
        message: Optional[str],
        *,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.active_task:
            return

        task = self.active_task
        merged_metadata = {"passed": passed, "message": message}
        if metadata:
            merged_metadata.update(metadata)

        self.log_event("checker_run", trace_id=trace_id, metadata=merged_metadata)
        if passed:
            if task.first_pass_turn is None:
                task.first_pass_turn = task.number_of_generate_calls
            task.final_checker_message = message
            self.log_event("checker_passed", trace_id=trace_id, metadata=merged_metadata)
        else:
            task.checker_fail_count += 1
            task.final_checker_message = message
            self.log_event("checker_failed", trace_id=trace_id, metadata=merged_metadata)

    def finish_task(
        self,
        *,
        success: bool,
        final_checker_message: Optional[str] = None,
        failure_reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.active_task:
            return None

        task = self.active_task
        task.task_end_timestamp = _utc_now_iso()
        task.success = success
        task.failure_reason = failure_reason
        if final_checker_message is not None:
            task.final_checker_message = final_checker_message

        finish_metadata = metadata or {}
        if failure_reason:
            finish_metadata = {**finish_metadata, "failure_reason": failure_reason}
        self.log_event("task_finished" if success else "task_abandoned", metadata=finish_metadata)

        summary = {
            "participant_id": task.participant_id,
            "session_id": task.session_id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "dataset_name": task.dataset_name,
            "model_name": task.model_name,
            "condition": task.condition,
            "task_start_timestamp": task.task_start_timestamp,
            "task_end_timestamp": task.task_end_timestamp,
            "wall_clock_seconds": self._seconds_since(task.task_start_timestamp, task.task_end_timestamp),
            "generation_time_seconds": task.generation_time_seconds,  # G2
            "success": task.success,
            "number_of_generate_calls": task.number_of_generate_calls,
            "number_of_correction_generate_calls": task.number_of_correction_generate_calls,
            "total_generated_tokens": task.total_generated_tokens,
            "correction_tokens": task.correction_tokens,
            "final_turn_index": task.final_turn_index,
            "first_pass_turn": task.first_pass_turn,
            "checker_fail_count": task.checker_fail_count,
            "final_checker_message": task.final_checker_message,
            "total_action_count": task.total_action_count,
            "total_clicks": task.total_clicks,
            "total_probes": task.total_probes,
            "total_branch_actions": task.total_branch_actions,
            "total_user_messages": task.total_user_messages,
            "total_user_characters": task.total_user_characters,
            "total_user_prompt_tokens": task.total_user_prompt_tokens,
            "trace_id": task.trace_id,
            "failure_reason": task.failure_reason,
            "metadata": json.dumps(task.metadata, ensure_ascii=False),
        }
        self._append_jsonl(self.task_summary_path, summary)
        self._append_task_summary_csv(summary)
        self.active_task = None
        return summary

    @staticmethod
    def _seconds_since(start_iso: str, end_iso: str) -> float:
        start_dt = datetime.fromisoformat(start_iso)
        end_dt = datetime.fromisoformat(end_iso)
        return max((end_dt - start_dt).total_seconds(), 0.0)