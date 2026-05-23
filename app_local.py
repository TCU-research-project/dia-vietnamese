import re
import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from dia.model import Dia
from dia.config import DiaConfig
from dia.layers import DiaModel
import dac
import safetensors.torch as st
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file  

# --- Patch PyTorch 2.6: đảm bảo torch.load không dùng weights_only=True mặc định ---
_orig_torch_load = torch.load
def _torch_load_compat(path, *args, **kwargs):
    """
    Load checkpoint tương thích cả .pt/.pth và .safetensors
    """
    if isinstance(path, str) and path.endswith(".safetensors"):
        try:
            return st.load_file(path)
        except RuntimeError as e:
            msg = str(e)
            if "unable to mmap" in msg or "Cannot allocate memory" in msg:
                raise RuntimeError(
                    "Unable to memory-map the safetensors checkpoint. "
                    "This usually means WSL does not have enough RAM/swap for the 6.44GB model, "
                    "or the checkpoint is being loaded from a Windows-mounted path like /mnt/c or /mnt/d. "
                    "Move the repo/checkpoint into the WSL Linux filesystem (for example ~/dia-vietnamese) "
                    "and increase WSL memory/swap in %USERPROFILE%\\.wslconfig."
                ) from e
            raise
    else:
        return _orig_torch_load(path, *args, **kwargs)
torch.load = _torch_load_compat


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def normalize_state_key(key: str) -> str:
    key = key.replace("module.", "")
    if key.startswith("model."):
        key = key[6:]
    return key


def load_safetensors_streaming(module: torch.nn.Module, checkpoint_path: str, strict: bool = True) -> None:
    """
    Load safetensors one tensor at a time to avoid materializing the full state_dict in RAM.
    This is important for WSL machines with small system memory.
    """
    target_state = module.state_dict()
    missing = set(target_state.keys())
    unexpected: list[str] = []

    try:
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for raw_key in f.keys():
                key = normalize_state_key(raw_key)
                if key not in target_state:
                    unexpected.append(raw_key)
                    continue

                tensor = f.get_tensor(raw_key)
                target = target_state[key]
                if tensor.shape != target.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {key}: checkpoint {tuple(tensor.shape)} != model {tuple(target.shape)}"
                    )

                target.copy_(tensor.to(dtype=target.dtype))
                missing.discard(key)
                del tensor
    except RuntimeError as e:
        msg = str(e)
        if "unable to mmap" in msg or "Cannot allocate memory" in msg:
            raise RuntimeError(
                "Unable to memory-map the safetensors checkpoint. "
                "On an 8GB RAM WSL machine, move the project to the WSL Linux filesystem "
                "(for example ~/dia-vietnamese), set a large WSL swap in %USERPROFILE%\\.wslconfig, "
                "and run with --device cuda --half True."
            ) from e
        raise

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Error loading checkpoint: missing={len(missing)} unexpected={len(unexpected)}"
        )

    print(f"[load] streaming safetensors complete: missing={len(missing)} unexpected={len(unexpected)}")


def prepare_runtime_config(cfg: DiaConfig, use_half: bool, runtime_device: torch.device) -> DiaConfig:
    """
    Keep activation and weight precision aligned with the runtime model.

    `config_inference.json` is fp32 by default, but CUDA inference can cast the
    module weights to fp16. If the config still asks layers to emit fp32
    activations, cross-attention can build fp32 K/V caches while decoder queries
    are fp16, which breaks scaled_dot_product_attention.
    """
    if not (use_half and runtime_device.type == "cuda"):
        return cfg

    return cfg.model_copy(
        update={
            "training": cfg.training.model_copy(update={"dtype": "float16"}),
            "model": cfg.model.model_copy(update={"weight_dtype": "float16"}),
        }
    )


def load_env_file(env_path: str = ".env") -> None:
    """
    Load simple KEY=VALUE entries from a .env file without adding a dependency.
    Existing environment variables are kept unchanged.
    """
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def login_hugging_face_from_env(env_path: str = ".env") -> str | None:
    load_env_file(env_path)
    token = get_hf_token()
    if not token:
        return None

    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)
    print(f"Logged in to Hugging Face with token from {env_path}")
    return token


def ensure_local_checkpoint(args) -> None:
    ckpt_path = Path(args.local_ckpt)
    config_path = Path(args.config)

    token = login_hugging_face_from_env(args.env_file)
    if ckpt_path.exists() and config_path.exists():
        return

    if args.no_auto_download:
        raise RuntimeError(
            f"Checkpoint not found: {ckpt_path}. "
            "Remove --no-auto-download or download model.safetensors manually."
        )
    if not token:
        raise RuntimeError(
            f"Checkpoint not found: {ckpt_path}. Add HF_TOKEN=hf_... to {args.env_file}, "
            "accept the gated model on Hugging Face, then run again."
        )

    from huggingface_hub import snapshot_download

    local_dir = ckpt_path.parent if ckpt_path.parent != Path("") else Path(".")
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo_id} to {local_dir}...")
    snapshot_download(
        args.repo_id,
        local_dir=local_dir,
        repo_type="model",
        token=token,
        allow_patterns=["model.safetensors", "config_inference.json"],
    )

    if not ckpt_path.exists():
        raise RuntimeError(f"Download finished, but checkpoint is still missing: {ckpt_path}")
    if not config_path.exists():
        raise RuntimeError(f"Download finished, but config is still missing: {config_path}")


# Textbox để hiển thị trạng thái load model
status = gr.Textbox(label="Model Status", interactive=False)

# Đặt global ở gần đầu file (nếu chưa có)
model = None
dac_model = None

def load_model_once():
    """
    Load model duy nhất một lần khi khởi động.
    Giữ logic .safetensors -> .pt, half/compile chỉ trên CUDA, gắn DAC một lần.
    """
    global model, dac_model

    if model is not None:
        return f"Model already loaded on {device}"

    ckpt_path = Path(args.local_ckpt)
    tmp_pt_path = None

    # Nếu checkpoint là .safetensors, chuyển tạm sang .pt để tương thích torch.load
    if ckpt_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as safe_load_file
        except Exception as e:
            raise RuntimeError("Chưa cài safetensors, không thể nạp .safetensors") from e

        state_dict = safe_load_file(str(ckpt_path), device="cpu")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        tmp_pt_path = tmp.name
        tmp.close()
        torch.save(state_dict, tmp_pt_path)
        ckpt_to_load = tmp_pt_path
    else:
        ckpt_to_load = str(ckpt_path)

    # Gọi Dia.from_local với compute_dtype nếu bạn đã khai báo ở Bước 2
    kwargs = dict(
        config_path=args.config,
        checkpoint_path=ckpt_to_load,
        device=device,
    )
    if "compute_dtype" in globals():
        kwargs["compute_dtype"] = compute_dtype

    model_local = Dia.from_local(**kwargs)

    # Xoá file tạm (nếu có)
    if tmp_pt_path is not None:
        try:
            Path(tmp_pt_path).unlink(missing_ok=True)
        except Exception:
            pass

    # half / compile CHỈ trên CUDA
    if getattr(args, "half", False) and device.type == "cuda" and hasattr(model_local, "model"):
        model_local.model = model_local.model.half()

    if getattr(args, "compile", False) and device.type == "cuda" and hasattr(model_local, "model"):
        model_local.model = torch.compile(model_local.model, backend="inductor")

    # Gắn DAC đúng device — chỉ load một lần cho toàn app
    if dac_model is None:
        _dac = dac.DAC.load(dac.utils.download()).to(device)
        dac_model_local = _dac
        globals()["dac_model"] = dac_model_local
    else:
        dac_model_local = dac_model

    model_local.dac_model = dac_model_local

    # Xuất ra global
    model = model_local
    return f"Loaded checkpoint: {ckpt_path.name} on {device}"

# --- Global Setup ---
parser = argparse.ArgumentParser(description="Gradio interface for Nari TTS")
parser.add_argument("--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
parser.add_argument("--local_ckpt", type=str, default="dia/model.safetensors", help="path to your local checkpoint")
parser.add_argument("--config", type=str, default="dia/config_inference.json", help="path to your inference")
parser.add_argument("--repo-id", type=str, default="cosrigel/dia-finetuning-vnese", help="Hugging Face model repo")
parser.add_argument("--env-file", type=str, default=".env", help="path to .env file containing HF_TOKEN")
parser.add_argument("--no-auto-download", action="store_true", help="do not download checkpoint automatically")
parser.add_argument("--half", type=parse_bool, nargs="?", const=True, default=True, help="load model in fp16 on CUDA")
parser.add_argument("--no-half", dest="half", action="store_false", help="disable fp16 model loading")
parser.add_argument("--compile", type=parse_bool, nargs="?", const=True, default=False, help="torch compile model")

args = parser.parse_args()
ensure_local_checkpoint(args)

# Determine device
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    try:
        # Cho phép TF32: nhanh hơn trên Ampere+ mà vẫn ổn định
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        # Ưu tiên Flash-Attention trong SDPA nếu có
        from torch.nn.attention import sdpa_kernel
        sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    except Exception:
        pass
        
# Dtype cho Dia (app.py dùng chuỗi cho compute_dtype)
_dtype_map = {"cpu": "float32", "mps": "float32", "cuda": "float16"}
compute_dtype = _dtype_map.get(device.type, "float16")
print(f"compute_dtype for Dia: {compute_dtype}")

# Load Nari model and config
print("Loading Nari model...")

try:
    cfg = DiaConfig.load(args.config if getattr(args, "config", None) else "dia/config.json")
    cfg = prepare_runtime_config(cfg, getattr(args, "half", False), device)

    ptmodel = DiaModel(cfg)

    # ✅ half chỉ trên CUDA
    if getattr(args, "half", False) and device.type == "cuda":
        ptmodel = ptmodel.half()

    # ✅ compile chỉ trên CUDA
    if getattr(args, "compile", False) and device.type == "cuda":
        ptmodel = torch.compile(ptmodel, backend="inductor")

    # Tải checkpoint ít peak RAM nhất có thể.
    if str(args.local_ckpt).endswith(".safetensors"):
        load_safetensors_streaming(ptmodel, args.local_ckpt, strict=True)
    else:
        state = _torch_load_compat(args.local_ckpt, map_location="cpu")
        ptmodel.load_state_dict(state["model"] if "model" in state else state, strict=True)
        del state

    print("✅ Model loaded successfully! Please wait...")
    ptmodel = ptmodel.to(device).eval()

    model = Dia(cfg, device)
    model.model = ptmodel

    # ✅ DAC đúng device
    dac_model = dac.DAC.load(dac.utils.download()).to(device)
    model.dac_model = dac_model

except Exception as e:
    print(f"Error loading Nari model: {e}")
    raise

def trim_silence(audio: np.ndarray, threshold: float = 0.01, margin: int = 1000) -> np.ndarray:
    """
    Cắt bỏ vùng im lặng ở đầu và cuối audio numpy.
    - `threshold`: ngưỡng biên độ để coi là 'có tiếng'
    - `margin`: giữ lại một ít trước và sau vùng có tiếng (tính theo mẫu)
    """
    abs_audio = np.abs(audio)
    non_silent_indices = np.where(abs_audio > threshold)[0]

    if non_silent_indices.size == 0:
        return audio  # Nếu hoàn toàn im lặng

    start = max(non_silent_indices[0] - margin, 0)
    end = min(non_silent_indices[-1] + margin, len(audio))

    return audio[start:end]

def run_inference(
    text_input: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_new_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    speed_factor: float,
):


    print(f"[DEBUG] max_new_tokens = {max_new_tokens}")
    """
    Runs Nari inference using the globally loaded model and provided inputs.
    Uses temporary files for text and audio prompt compatibility with inference.generate.
    """
    global model, device  # Access global model, config, device
    # ✅ Reset conditioning cache nếu có
    if hasattr(model, "reset_conditioning"):
        model.reset_conditioning()
        print("[DEBUG] Đã reset conditioning latent voice.")
    elif hasattr(model, "voice_encoder_cache"):
        model.voice_encoder_cache = {}
        print("[DEBUG] Đã xoá voice encoder cache.")
    else:
        print("[DEBUG] Không tìm thấy cơ chế reset conditioning, bỏ qua.")


    if not text_input or text_input.isspace():
        raise gr.Error("Text input cannot be empty.")

    temp_txt_file_path = None
    temp_audio_prompt_path = None
    output_audio = (44100, np.zeros(1, dtype=np.float32))

    try:
        prompt_path_for_generate = None
        
        if audio_prompt_input is not None:
            sr, audio_data = audio_prompt_input
            # Resample nếu không phải 44100
            if sr != 44100:
                try:
                    import librosa
                    # librosa yêu cầu float32 input
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=44100)
                    sr = 44100
                except Exception as e:
                    raise gr.Error(f"Resampling failed: {e}")
                
            # Check if audio_data is valid
            if (
                audio_data is None
                or audio_data.size == 0
                or np.max(np.abs(audio_data)) < 1e-4  # quá nhỏ
                or len(audio_data) < 1000             # quá ngắn (tương đương ~23ms ở 44.1kHz)
            ):
                gr.Warning("Audio prompt quá ngắn hoặc không hợp lệ sau xử lý. Đã bỏ qua prompt.")
                audio_prompt_input = None
                prompt_path_for_generate = None
                temp_audio_prompt_path = None
            else:
                # Save prompt audio to a temporary WAV file
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".wav", delete=False
                ) as f_audio:
                    temp_audio_prompt_path = f_audio.name  # Store path for cleanup

                    # Basic audio preprocessing for consistency
                    # Convert to float32 in [-1, 1] range if integer type
                    if np.issubdtype(audio_data.dtype, np.integer):
                        max_val = np.iinfo(audio_data.dtype).max
                        audio_data = audio_data.astype(np.float32) / max_val
                    elif not np.issubdtype(audio_data.dtype, np.floating):
                        gr.Warning(
                            f"Unsupported audio prompt dtype {audio_data.dtype}, attempting conversion."
                        )
                        # Attempt conversion, might fail for complex types
                        try:
                            audio_data = audio_data.astype(np.float32)
                        except Exception as conv_e:
                            raise gr.Error(
                                f"Failed to convert audio prompt to float32: {conv_e}"
                            )

                    # Ensure mono (average channels if stereo)
                    if audio_data.ndim > 1:
                        if audio_data.shape[0] == 2:  # Assume (2, N)
                            audio_data = np.mean(audio_data, axis=0)
                        elif audio_data.shape[1] == 2:  # Assume (N, 2)
                            audio_data = np.mean(audio_data, axis=1)
                        else:
                            gr.Warning(
                                f"Audio prompt has unexpected shape {audio_data.shape}, taking first channel/axis."
                            )
                            audio_data = (
                                audio_data[0]
                                if audio_data.shape[0] < audio_data.shape[1]
                                else audio_data[:, 0]
                            )
                        audio_data = np.ascontiguousarray(
                            audio_data
                        )  # Ensure contiguous after slicing/mean

                    # Write using soundfile
                    try:
                        sf.write(
                            temp_audio_prompt_path, audio_data, sr, subtype="FLOAT"
                        )  # Explicitly use FLOAT subtype
                        prompt_path_for_generate = temp_audio_prompt_path
                        print(
                            f"Created temporary audio prompt file: {temp_audio_prompt_path} (orig sr: {sr})"
                        )
                    except Exception as write_e:
                        print(f"Error writing temporary audio file: {write_e}")
                        raise gr.Error(f"Failed to save audio prompt: {write_e}")
                    

        # 3. Run Generation

        start_time = time.time()

        # Use torch.inference_mode() context manager for the generation call
        # 3. Xử lý văn bản dài bằng cách tách câu
        # --- Nếu CÓ audio prompt: xử lý nguyên khối, không chia câu ---
        if prompt_path_for_generate:
            chunks = [text_input.strip()]
            print("[INFO] Đã phát hiện audio prompt - xử lý toàn bộ văn bản như một đoạn duy nhất.")
        else:
            # --- Nếu KHÔNG có audio prompt: chia theo speaker và câu như bình thường ---
            speaker_blocks = re.split(r'(?=\[[^\]]+\])', text_input.strip())
            chunks = []
            current_speaker = None
        
            for block in speaker_blocks:
                block = block.strip()
                if not block:
                    continue
        
                speaker_match = re.match(r"\[([^\]]+)\]\s*(.*)", block, re.DOTALL)
                if speaker_match:
                    current_speaker = speaker_match.group(1)
                    content = speaker_match.group(2).strip()
                else:
                    content = block
        
                sentences = re.split(r'(?<=[.!?])\s+', content)
                for sent in sentences:
                    sent = sent.strip()
                    if sent:
                        if current_speaker:
                            chunks.append(f"[{current_speaker}] {sent}")
                        else:
                            chunks.append(sent)
            print(f"[INFO] Văn bản được chia thành {len(chunks)} đoạn theo speaker/câu.")   
        
        # Sinh từng đoạn nhỏ và nối lại
        generated_segments = []
        with torch.inference_mode():
            print(f"📄 Văn bản dài, tách thành {len(chunks)} đoạn.")
            for idx, chunk in enumerate(chunks):
                print(f"[Đoạn {idx+1}] {chunk}")
                
                text_for_model = chunk  # channel đã nằm trong chunk rồi, không cần thêm
        
                segment = model.generate(
                    text_for_model,
                    max_tokens=max_new_tokens,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    use_cfg_filter=True,
                    cfg_filter_top_k=cfg_filter_top_k,
                    use_torch_compile=False,
                    audio_prompt_path=prompt_path_for_generate,
                )
                if segment is not None and isinstance(segment, np.ndarray):
                    segment = trim_silence(segment, threshold=0.01, margin=1000)
                    # ✅ Thêm khoảng nghỉ ngắn vào cuối mỗi câu để nghe giống người
                    pause = np.zeros(int(0.5 * 44100), dtype=np.float32)  # 0.25s pause
                    segment = np.concatenate([segment, pause])
                    generated_segments.append(segment)
        
        # Ghép toàn bộ đoạn lại (có thể thêm silence nếu cần)
        if generated_segments:
            combined = []
            group = []
            for i, seg in enumerate(generated_segments):
                group.append(seg)
                if len(group) == 2 or i == len(generated_segments) - 1:
                    # Ghép 2 câu lại thành 1 đoạn
                    if len(group) == 2:
                        merged = np.concatenate(group)
                    else:
                        merged = group[0]
                    combined.append(merged)
                    group = []
            output_audio_np = np.concatenate(combined)

        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")

        # 4. Convert Codes to Audio
        if output_audio_np is not None:
            # Get sample rate from the loaded DAC model
            output_sr = 44100

            # --- Slow down audio ---
            original_len = len(output_audio_np)
            # Ensure speed_factor is positive and not excessively small/large to avoid issues
            speed_factor = max(0.1, min(speed_factor, 5.0))
            target_len = int(
                original_len / speed_factor
            )  # Target length based on speed_factor
            if (
                target_len != original_len and target_len > 0
            ):  # Only interpolate if length changes and is valid
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
                output_audio = (
                    output_sr,
                    resampled_audio_np.astype(np.float32),
                )  # Use resampled audio
                print(
                    f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed."
                )
            else:
                output_audio = (
                    output_sr,
                    output_audio_np,
                )  # Keep original if calculation fails or no change
                print(f"Skipping audio speed adjustment (factor: {speed_factor:.2f}).")
            # --- End slowdown ---

            print(
                f"Audio conversion successful. Final shape: {output_audio[1].shape}, Sample Rate: {output_sr}"
            )

        else:
            print("\nGeneration finished, but no valid tokens were produced.")
            # Return default silence
            gr.Warning("Generation produced no output.")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()
        # Re-raise as Gradio error to display nicely in the UI
        raise gr.Error(f"Inference failed: {e}")

    finally:
        # 5. Cleanup Temporary Files defensively
        if temp_txt_file_path and Path(temp_txt_file_path).exists():
            try:
                Path(temp_txt_file_path).unlink()
                print(f"Deleted temporary text file: {temp_txt_file_path}")
            except OSError as e:
                print(
                    f"Warning: Error deleting temporary text file {temp_txt_file_path}: {e}"
                )
        if temp_audio_prompt_path and Path(temp_audio_prompt_path).exists():
            try:
                Path(temp_audio_prompt_path).unlink()
                print(f"Deleted temporary audio prompt file: {temp_audio_prompt_path}")
            except OSError as e:
                print(
                    f"Warning: Error deleting temporary audio prompt file {temp_audio_prompt_path}: {e}"
                )

    return output_audio

# --- Create Gradio Interface ---
css = """
#col-container {max-width: 90%; margin-left: auto; margin-right: auto;}
"""
# Attempt to load default text from example.txt
default_text = "[KienThucQuanSu] Thay thế bằng đoạn văn bản cần sinh giọng nói."
example_txt_path = Path("./example.txt")
if example_txt_path.exists():
    try:
        default_text = example_txt_path.read_text(encoding="utf-8").strip()
        if not default_text:  # Handle empty example file
            default_text = "Example text file was empty."
    except Exception as e:
        print(f"Warning: Could not read example.txt: {e}")

# Build Gradio UI
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Nari Text-to-Speech Synthesis")

    # Chỉ hiển thị trạng thái (không còn dropdown chọn checkpoint)
    with gr.Row():
        status.render()

    # Load model duy nhất một lần khi khởi động
    init_msg = load_model_once()
    status.value = init_msg
    
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text here...",
                value=default_text,
                lines=5,  # Increased lines
            )
            audio_prompt_input = gr.Audio(
                label="Audio Prompt (Optional)",
                show_label=True,
                sources=["upload", "microphone"],
                type="numpy",
            )
            with gr.Accordion("Generation Parameters", open=False):
                max_new_tokens = gr.Slider(
                    label="Max New Tokens (Audio Length)",
                    minimum=860,
                    maximum=3072,
                    value=3072,  # Use config default if available, else fallback
                    step=50,
                    info="Controls the maximum length of the generated audio (more tokens = longer audio).",
                )
                cfg_scale = gr.Slider(
                    label="CFG Scale (Guidance Strength)",
                    minimum=1.0,
                    maximum=5.0,
                    value=3.0,  # Default from inference.py
                    step=0.1,
                    info="Higher values increase adherence to the text prompt.",
                )
                temperature = gr.Slider(
                    label="Temperature (Randomness)",
                    minimum=1.0,
                    maximum=1.5,
                    value=1.3,  # Default from inference.py
                    step=0.05,
                    info="Lower values make the output more deterministic, higher values increase randomness.",
                )
                top_p = gr.Slider(
                    label="Top P (Nucleus Sampling)",
                    minimum=0.80,
                    maximum=1.0,
                    value=0.95,  # Default from inference.py
                    step=0.01,
                    info="Filters vocabulary to the most likely tokens cumulatively reaching probability P.",
                )
                cfg_filter_top_k = gr.Slider(
                    label="CFG Filter Top K",
                    minimum=15,
                    maximum=50,
                    value=35,
                    step=1,
                    info="Top k filter for CFG guidance.",
                )
                speed_factor_slider = gr.Slider(
                    label="Speed Factor",
                    minimum=0.8,
                    maximum=1.0,
                    value=0.94,
                    step=0.02,
                    info="Adjusts the speed of the generated audio (1.0 = original speed).",
                )

            run_button = gr.Button("Generate Audio", variant="primary")
        #
        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Audio",
                type="numpy",
                autoplay=False,
            )
            
            gr.Markdown("📌 **Copy tag người nói như `[KienThucQuanSu]` để dán vào văn bản sinh giọng phù hợp.**")
            
            gr.Markdown("### 🟢 Good Voice Speakers (Rõ, chuẩn, chất lượng cao)")
            gr.Dataframe(
                headers=["North Male", "North Female", "South Male", "South Female", "Center Female"],
                value=[
                    ["[KienThucQuanSu]", "[kenhCoVan]", "[HocEnglishOnline]", "[CoBaBinhDuong]", "[PTTH-TRT]"],
                    ["[AnimeRewind.Official]", "[ThePresentWriter]", "[HuynhDuyKhuongofficial]", "[SUCKHOETAMSINH]", ""],
                    ["[BroNub]", "[5PhutCrypto]", "[HuynhLapOfficial]", "[TIN3PHUT]", ""],
                    ["[VuiVe]", "[SachBiQuyetThanhCong]", "[NgamRadio]", "", ""],
                    ["[W2WAnime]", "[BIBITV8888]", "", "", ""],
                    ["[DongMauViet]", "", "", "", ""],
                ],
                interactive=False
            )
            
            gr.Markdown("### 🟡 Normal Voice Speakers (Dùng được, giọng khá ổn)")
            gr.Dataframe(
                headers=["North Male", "North Female", "South Male", "South Female"],
                value=[
                    ["[NhaNhac555]", "[sunhuynpodcast.]", "[MensBay]", "[BoringPPL]"],
                    ["[JVevermind]", "[HocvienBovaGau]", "[Web5Ngay]", "[TULEMIENTAY]"],
                    ["[CosmicWriter]", "[SukiesKitchen]", "[AnhBanThan]", "[HappyHidari]"],
                    ["[RuaNgao]", "[Nhantaidaiviet]", "[PhanTichGame]", "[SpiderumBooks]"],
                    ["[TuanTienTi2911]", "[W2WCartoon]", "", "[HoabinhTVgo]"],
                    ["[CuThongThai]", "[BaodientuVOV]", "", "[RiwayLegal]"],
                    ["[meGAME_Official]", "", "", ""],
                ],
                interactive=False
            )
            
            gr.Markdown("### 🔴 Weak Voice Speakers (Không nên ưu tiên dùng làm mẫu giọng)")
            gr.Dataframe(
                headers=["North Male", "North Female", "South Male", "South Female"],
                value=[
                    ["[TintucBitcoin247]", "[Xanh24h]", "[MangoVid]", "[TheGioiLaptop]"],
                    ["[ThanhPahm]", "", "[ThaiNhiTV]", "[BachHoaXANHcom]"],
                    ["[VuTruNguyenThuy]", "", "[MeovatcuocsongLNV]", ""],
                    ["[NTNVlogsNguyenThanhNam]", "", "", ""],
                    ["[HIEUROTRONG5PHUT-NTKT]", "", "", ""],
                ],
                interactive=False
            )

    # Link button click to function
    run_button.click(
        fn=run_inference,
        inputs=[
            text_input,
            audio_prompt_input,
            max_new_tokens,
            cfg_scale,
            temperature,
            top_p,
            cfg_filter_top_k,
            speed_factor_slider,
        ],
        outputs=[audio_output],  # Add status_output here if using it
        api_name="generate_audio",
    )

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(share=args.share, server_name="0.0.0.0")
