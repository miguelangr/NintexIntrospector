import os
from huggingface_hub import snapshot_download

# Configurar el mirror de Hugging Face
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Configurar el proxy (ajusta la URL y el puerto según tu configuración)
proxies = {
    "http": "http://proxy.tuempresa.com:puerto",
    "https": "http://proxy.tuempresa.com:puerto"
}

# Intentar descargar el modelo Mistral
try:
    modelo_path = snapshot_download(
        repo_id="TheBloke/Mistral-7B-v0.1-GGML",
        repo_type="model",
        proxies=proxies,
        max_workers=4  # Ajusta este número según tu conexión
    )
    print(f"Modelo Mistral descargado exitosamente en: {modelo_path}")
except Exception as e:
    print(f"Error al descargar el modelo: {e}")
    # Si falla, intenta sin proxy
    try:
        modelo_path = snapshot_download(
            repo_id="TheBloke/Mistral-7B-v0.1-GGML",
            repo_type="model",
            max_workers=4
        )
        print(f"Modelo Mistral descargado exitosamente sin proxy en: {modelo_path}")
    except Exception as e:
        print(f"Error al descargar el modelo sin proxy: {e}")

