venv\Scripts\activate.bat
python -c “from huggingface_hub.hf_api import HfFolder; HfFolder.save_token(‘my_token’)”