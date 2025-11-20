#!/usr/bin/env python3
"""
Check transformers version compatibility for MERaLiON model.
"""

def check_transformers_version():
    """Check if transformers version is compatible with MERaLiON."""
    try:
        import transformers
        version = transformers.__version__
        print(f"📦 Current transformers version: {version}")
        
        if version == "4.50.1":
            print("✅ Transformers version is correct for MERaLiON model")
            return True
        else:
            print(f"⚠️  Transformers version mismatch!")
            print(f"   Required: 4.50.1")
            print(f"   Current: {version}")
            print("")
            print("🔧 To fix this issue:")
            print("   poetry run pip install transformers==4.50.1")
            print("   or")
            print("   make clean && make build")
            return False
            
    except ImportError:
        print("❌ Transformers not installed")
        print("🔧 To install:")
        print("   poetry install")
        return False

def check_other_dependencies():
    """Check other required dependencies."""
    dependencies = [
        ("torch", "PyTorch", True),
        ("librosa", "Librosa", True),
        ("soundfile", "SoundFile", True),
        ("safetensors", "SafeTensors", True),
        ("accelerate", "Accelerate", False),  # Optional but recommended
    ]
    
    print("\n🔍 Checking other dependencies:")
    all_required_good = True
    
    for module, name, required in dependencies:
        try:
            __import__(module)
            status = "✅" if required else "✅ (optional)"
            print(f"{status} {name}: OK")
        except ImportError:
            if required:
                print(f"❌ {name}: Missing (REQUIRED)")
                all_required_good = False
            else:
                print(f"⚠️  {name}: Missing (optional but recommended)")
    
    return all_required_good

def main():
    """Main function."""
    print("=" * 60)
    print("🔍 MERaLiON Model Compatibility Check")
    print("=" * 60)
    
    transformers_ok = check_transformers_version()
    deps_ok = check_other_dependencies()
    
    print("\n" + "=" * 60)
    if transformers_ok and deps_ok:
        print("🎉 All dependencies are compatible!")
        print("You can now run: make download")
    else:
        print("⚠️  Some dependencies need to be fixed")
        print("Run: poetry install")
        print("Then: make download")
    print("=" * 60)

if __name__ == "__main__":
    main()
