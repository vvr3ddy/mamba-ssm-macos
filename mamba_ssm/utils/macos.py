import json
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from transformers import AutoTokenizer

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def create_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_config_file(config_path):
    if not Path(config_path).exists():
        return None
    with open(config_path) as f:
        return json.load(f)


def create_mamba1_config(config_data):
    if "ssm_cfg" not in config_data:
        config_data["ssm_cfg"] = {}
    if "layer" not in config_data["ssm_cfg"]:
        config_data["ssm_cfg"]["layer"] = "Mamba1"
    return MambaConfig(**config_data)


def create_mamba2_config(config_data):
    config_data["ssm_cfg"] = {
        "layer": "Mamba2",
        "d_state": 128,
        "d_conv": 4,
        "expand": 2,
        "headdim": 64,
        "ngroups": 1,
    }
    return MambaConfig(**config_data)


def get_optimal_dtype(device: str) -> torch.dtype:
    """
    Get optimal dtype for the given device.
    
    Args:
        device: Device string ("mps", "cuda", "cpu")
    
    Returns:
        Optimal torch.dtype for the device
    """
    if device == "mps":
        # bfloat16 available on M2+ (MPS backend >= PyTorch 2.1)
        # float16 works on all Apple Silicon
        try:
            _ = torch.zeros(1, dtype=torch.bfloat16, device=device)
            return torch.bfloat16
        except RuntimeError:
            return torch.float16
    elif device == "cuda":
        return torch.bfloat16
    return torch.float32


def load_and_prepare_model(
    model_name: str,
    model_dir: str,
    device: str,
    dtype: Optional[torch.dtype] = None,
    compile_model: bool = False,
) -> Tuple[bool, Optional[MambaLMHeadModel], Optional[AutoTokenizer]]:
    """
    Load and prepare a Mamba model for inference.
    
    Args:
        model_name: Name of the model ("mamba1" or "mamba2")
        model_dir: Directory containing model files
        device: Device to load model on ("mps", "cuda", "cpu")
        dtype: Optional dtype override (defaults to optimal for device)
        compile_model: Whether to compile the model with torch.compile
    
    Returns:
        Tuple of (success, model, tokenizer)
    """
    config_path = f"{model_dir}/{model_name}/{model_name}-130m-config.json"
    weight_path = f"{model_dir}/{model_name}/{model_name}-130m-model.bin"
    config_creator = create_mamba1_config if model_name == "mamba1" else create_mamba2_config

    config_data = load_config_file(config_path)
    if not config_data:
        print("❌ No config file found")
        return False, None, None

    print(f"Using config: {config_path}")
    config = config_creator(config_data)
    
    # Determine dtype
    if dtype is None:
        dtype = get_optimal_dtype(device)
    
    model = MambaLMHeadModel(config, device=device, dtype=dtype)
    tokenizer = create_tokenizer()

    if not Path(weight_path).exists():
        print("❌ No weight file found")
        return False, None, None

    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Optionally compile the model
    if compile_model and device == "mps":
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, backend="inductor", dynamic=True, fullgraph=False)
            print("🚀 Model compiled with torch.inductor for MPS acceleration")
        except Exception as e:
            print(f"⚠️ torch.compile failed, using eager mode: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🎯 Model ready: {total_params:,} parameters ({dtype.__name__})")
    return True, model, tokenizer


def generate_text_with_model(
    model: MambaLMHeadModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str,
    max_length: int,
    temperature: float = 0.8,
    seed: Optional[int] = None,
    use_cache: bool = True,
) -> str:
    """
    Generate text using the Mamba model with optional inference caching.
    
    Args:
        model: The Mamba model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt
        device: Device to run on
        max_length: Maximum length of generated text
        temperature: Sampling temperature (0 for greedy)
        seed: Optional random seed
        use_cache: Whether to use inference caching (O(L) vs O(L²) complexity)
    
    Returns:
        Generated text string
    """
    if seed is not None:
        torch.manual_seed(seed)
        if device == "mps":
            torch.mps.manual_seed(seed)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    batch_size, prompt_len = input_ids.shape
    
    with torch.no_grad():
        if use_cache:
            # Use inference caching for O(L) complexity
            return _generate_with_cache(
                model, tokenizer, input_ids, device, max_length, temperature
            )
        else:
            # Original O(L²) implementation
            generated = input_ids.clone()
            for _ in range(max_length - input_ids.shape[1]):
                logits = model(generated).logits[:, -1, :]
                if temperature > 0:
                    next_token = torch.multinomial(
                        torch.softmax(logits / temperature, dim=-1), num_samples=1
                    )
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def _generate_with_cache(
    model: MambaLMHeadModel,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    device: str,
    max_length: int,
    temperature: float,
) -> str:
    """
    Generate text with inference caching (O(L) complexity).
    
    This uses InferenceParams to maintain SSM state across generation steps,
    avoiding recomputation of the full context at each step.
    """
    batch_size, prompt_len = input_ids.shape
    max_gen_len = max_length
    
    # Initialize inference parameters
    inference_params = InferenceParams(
        max_seqlen=max_gen_len,
        max_batch_size=batch_size,
    )
    
    # Process prompt in one forward pass to initialize states
    logits = model(input_ids, inference_params=inference_params).logits
    inference_params.seqlen_offset = prompt_len
    
    # Generate tokens
    generated = input_ids.clone()
    
    while generated.shape[1] < max_length:
        # Get logits for last token only
        last_token = generated[:, -1:]
        logits = model(last_token, inference_params=inference_params).logits[:, -1, :]
        
        # Sample next token
        if temperature > 0:
            next_token = torch.multinomial(
                torch.softmax(logits / temperature, dim=-1), num_samples=1
            )
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        generated = torch.cat([generated, next_token], dim=1)
        inference_params.seqlen_offset += 1
        
        # Check for EOS
        if (next_token == tokenizer.eos_token_id).all():
            break
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)
