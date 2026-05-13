import torch
import torch.onnx
import os
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.action_recognizer import ActionRecognizerModel

def export():
    # 1. Path setup
    checkpoint_path = "saved_models/action_recognizer_best.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "saved_models/action_recognizer_last.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: No checkpoint found in saved_models/")
        return

    print(f"📦 Loading checkpoint from: {checkpoint_path}")
    
    # 2. Load the model (try 5 classes first)
    try:
        model = ActionRecognizerModel(num_classes=5)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # Handle different checkpoint formats
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("✅ Model loaded successfully (5 classes detected).")
    except Exception as e:
        print(f"⚠️ Could not load as 5-class model ({e}). Trying 2-class fallback...")
        try:
            model = ActionRecognizerModel(num_classes=2)
            model.load_state_dict(state_dict)
            print("✅ Model loaded successfully (2-class fallback).")
        except Exception as e2:
            print(f"❌ Critical Error: Could not load model architecture: {e2}")
            return

    model.eval()

    # 3. Create dummy input (1 sequence, 16 frames, 3 channels, 112x112 px)
    dummy_input = torch.randn(1, 16, 3, 112, 112)

    # 4. Export to ONNX
    onnx_path = "saved_models/suraksha_vision.onnx"
    print(f"🚀 Exporting to ONNX: {onnx_path} ...")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"✨ Success! ONNX model ready for frontend: {onnx_path}")

if __name__ == "__main__":
    export()
