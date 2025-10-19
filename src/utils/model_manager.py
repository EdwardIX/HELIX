import torch
import threading
from transformers import AutoModel

class ModelManagerWithCPUOffload:
    _models = {}       # {model_name: model}
    _ref_counts = {}   # {model_name: count}
    _device_map = {}   # {model_name: "cpu"/"cuda"}
    _lock = threading.RLock()  # Reentrant lock, ensure thread safety

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """
        Create model (if not loaded) and place on CPU.
        Do not increase ref_count.
        """
        with cls._lock:
            if model_name not in cls._models:
                print(f"[ModelManager] Loading model '{model_name}' on CPU")
                model = AutoModel.from_pretrained(model_name, **kwargs).to("cpu")
                cls._models[model_name] = model
                cls._ref_counts[model_name] = 0
                cls._device_map[model_name] = "cpu"
            return cls._models[model_name]

    @classmethod
    def load_model(cls, model_name: str, device="cuda"):
        """
        Get model:
          - If ref_count==0, means nobody is using it  move to device
          - ref_count += 1
        """
        with cls._lock:
            if model_name not in cls._models:
                raise ValueError(f"Model '{model_name}' not loaded. Call from_pretrained() first.")

            if cls._ref_counts[model_name] == 0 and cls._device_map[model_name] != device:
                print(f"[ModelManager] Moving model '{model_name}' from {cls._device_map[model_name]} to {device}")
                cls._models[model_name] = cls._models[model_name].to(device)
                cls._device_map[model_name] = device

            cls._ref_counts[model_name] += 1
            print(f"[ModelManager] Load '{model_name}', ref_count={cls._ref_counts[model_name]}")
            return cls._models[model_name]

    @classmethod
    def release_model(cls, model_name: str):
        """
        Release model:
          - ref_count -= 1
          - If ref_count==0  move model back to CPU
        """
        with cls._lock:
            if model_name not in cls._models:
                raise ValueError(f"[ModelManager] Model '{model_name}' not loaded.")

            cls._ref_counts[model_name] -= 1
            print(f"[ModelManager] Release '{model_name}', ref_count={cls._ref_counts[model_name]}")

            if cls._ref_counts[model_name] == 0 and cls._device_map[model_name] != "cpu":
                print(f"[ModelManager] Moving model '{model_name}' back to CPU")
                cls._models[model_name] = cls._models[model_name].to("cpu")
                cls._device_map[model_name] = "cpu"
                torch.cuda.empty_cache()
