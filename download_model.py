from huggingface_hub import snapshot_download
import os
import multiprocessing

def download_model(model_id, local_dir=None):
    num_cpus = min(multiprocessing.cpu_count(), 32)

    print(f"Num workers: {num_cpus}")
    
    try:
        local_dir_path = snapshot_download(
            repo_id=model_id,
            max_workers=num_cpus,
            allow_patterns=['*.json', '*.safetensors']
        )
        print(f"Model successfully downloaded to: {local_dir_path}")
        return local_dir_path
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.1-405B-Instruct-FP8"
    print(f"Downloading {model_id}")
    download_path = download_model(model_id)

