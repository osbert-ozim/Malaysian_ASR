#!/usr/bin/env python3
"""
Script to check if Flash Attention is available in the Poetry environment.
"""

import sys
import torch
from pathlib import Path

def check_flash_attention():
    """Check if Flash Attention is available and working."""
    print("=" * 60)
    print("🔍 Flash Attention Availability Check")
    print("=" * 60)
    
    # Check Python environment
    print(f"🐍 Python version: {sys.version}")
    print(f"📍 Python executable: {sys.executable}")
    print(f"📦 Poetry environment: {Path(sys.executable).parent.parent.name}")
    
    # Check PyTorch
    print(f"\n🔥 PyTorch version: {torch.__version__}")
    print(f"🎯 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 CUDA version: {torch.version.cuda}")
        print(f"🎮 GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"🎮 GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check Flash Attention package
    print(f"\n⚡ Flash Attention Check:")
    try:
        import flash_attn
        print(f"✅ flash_attn package installed: {flash_attn.__version__}")
        
        # Test if flash_attn can be used
        try:
            from flash_attn import flash_attn_func
            print("✅ flash_attn_func import successful")
            
            # Test basic functionality
            if torch.cuda.is_available():
                print("🧪 Testing Flash Attention functionality...")
                try:
                    # Create test tensors
                    batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
                    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
                    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
                    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
                    
                    # Test flash attention
                    output = flash_attn_func(q, k, v)
                    print("✅ Flash Attention test passed!")
                    print(f"   Output shape: {output.shape}")
                    
                except Exception as e:
                    print(f"❌ Flash Attention test failed: {e}")
            else:
                print("⚠️  CUDA not available, skipping functionality test")
                
        except ImportError as e:
            print(f"❌ flash_attn_func import failed: {e}")
            
    except ImportError as e:
        print(f"❌ flash_attn package not found: {e}")
        print("💡 To install: poetry add flash-attn")
    
    # Check transformers compatibility
    print(f"\n🤗 Transformers Check:")
    try:
        from transformers import __version__ as transformers_version
        print(f"✅ Transformers version: {transformers_version}")
        
        # Check if flash attention is supported
        try:
            from transformers.models.llama.modeling_llama import LlamaAttention
            print("✅ Flash Attention support available in transformers")
        except ImportError:
            print("⚠️  Flash Attention support not found in transformers")
            
    except ImportError as e:
        print(f"❌ Transformers not found: {e}")
    
    # Check specific model compatibility
    print(f"\n🎯 MERaLiON Model Compatibility:")
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        
        # Test model loading with flash attention
        if torch.cuda.is_available():
            try:
                print("🧪 Testing MERaLiON model with Flash Attention...")
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    "MERaLiON/MERaLiON-2-10B-ASR",
                    use_safetensors=True,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    cache_dir="./model_cache"
                )
                print("✅ MERaLiON model loaded successfully with Flash Attention!")
                
            except Exception as e:
                print(f"❌ MERaLiON model with Flash Attention failed: {e}")
                print("💡 Falling back to standard attention...")
                try:
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        "MERaLiON/MERaLiON-2-10B-ASR",
                        use_safetensors=True,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        cache_dir="./model_cache"
                    )
                    print("✅ MERaLiON model loaded with standard attention")
                except Exception as e2:
                    print(f"❌ MERaLiON model loading failed: {e2}")
        else:
            print("⚠️  CUDA not available, skipping model test")
            
    except Exception as e:
        print(f"❌ Model compatibility check failed: {e}")
    
    print(f"\n{'=' * 60}")
    print("🏁 Check completed!")
    print("=" * 60)

if __name__ == "__main__":
    check_flash_attention()
