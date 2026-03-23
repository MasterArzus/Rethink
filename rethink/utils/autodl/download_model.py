import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download as hf_snapshot_download
except Exception:
    hf_snapshot_download = None

try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
except Exception:
    ms_snapshot_download = None

def download_model(model_id, save_path, source="huggingface"):
    """
    Download model to a deterministic local directory.
    """
    save_path = str(Path(save_path).expanduser())
    print(f"--- Starting download for model: {model_id} ---")
    print(f"Target directory: {save_path}")

    try:
        if source == "modelscope":
            if ms_snapshot_download is None:
                raise RuntimeError("modelscope is not installed")
            local_model_path = ms_snapshot_download(
                model_id,
                local_dir=save_path,
                local_files_only=False,
            )
        else:
            if hf_snapshot_download is None:
                raise RuntimeError("huggingface_hub is not installed")
            local_model_path = hf_snapshot_download(
                repo_id=model_id,
                local_dir=save_path,
                local_dir_use_symlinks=False,
                resume_download=True,
            )

        print(f"--- Model downloaded successfully to: {local_model_path} ---")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download model {model_id}. Reason: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 download_model.py <model_id> <save_path> [huggingface|modelscope]")
        sys.exit(1)

    model_id_to_download = sys.argv[1]
    path_to_save = sys.argv[2]
    source_name = sys.argv[3] if len(sys.argv) == 4 else "huggingface"

    if not download_model(model_id_to_download, path_to_save, source=source_name):
        sys.exit(1)
