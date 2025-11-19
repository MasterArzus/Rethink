import sys
from modelscope.hub.snapshot_download import snapshot_download

def download_model(model_id, save_path):
    """
    使用 modelscope 的 snapshot_download 函数下载模型。
    这个函数会自动处理缓存、显示进度并支持断点续传。
    """
    print(f"--- Starting download for model: {model_id} ---")
    print(f"Target directory: {save_path}")
    
    try:
        # cache_dir 指定了顶层缓存目录，模型会下载到其下的一个子目录中
        local_model_path = snapshot_download(
            model_id,
            cache_dir=save_path,
            # revision="master"  # 可以指定模型版本，默认为最新
        )
        print(f"--- Model downloaded successfully to: {local_model_path} ---")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download model {model_id}. Reason: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 download_model.py <model_id> <save_path>")
        sys.exit(1)

    model_id_to_download = sys.argv[1]
    path_to_save = sys.argv[2]

    if not download_model(model_id_to_download, path_to_save):
        sys.exit(1)
