from huggingface_hub import snapshot_download

snapshot_download(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", local_dir = 'tinyllama')
#snapshot_download(repo_id="huggyllama/llama-13b", local_dir = 'llama-13b')
