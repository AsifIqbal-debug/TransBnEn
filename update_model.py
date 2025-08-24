"""Script to update the specialized name transliteration model."""

import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

# Path to the specialized name transliteration model
model_path = Path(__file__).resolve().parent / "models" / "name_transliteration_model_corrected.pkl"

def update_model():
    """Update the specialized name model with the correct mapping."""
    try:
        print(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print(f"Loaded father name transliteration model with {len(model['direct_mappings'])} mappings")
            
            # Update the individual word mapping for "অম্বী"
            old_value = model['direct_mappings']["অম্বী"]
            print(f"Current mapping for 'অম্বী': {old_value}")
            
            # Update to "Ombi" to match NAME_MAPPINGS
            model['direct_mappings']["অম্বী"] = "Ombi"
            print(f"Updated mapping for 'অম্বী' to 'Ombi'")
            
            # Save the updated model
            with open(model_path, 'wb') as out_f:
                pickle.dump(model, out_f)
            print(f"Model saved to {model_path}")
            
    except Exception as e:
        print(f"Error updating model: {e}")

if __name__ == "__main__":
    update_model()
