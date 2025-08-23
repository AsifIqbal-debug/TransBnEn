"""Fine-tune the translation model with custom name pairs.

This script helps train the model to recognize specific name translations
like 'আসিফ -> Asif' that might not be handled properly by the default model.
"""

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# Configuration
TRAINING_DATA = "custom_names.csv"
MODEL_NAME = os.environ.get("INDIGO_MODEL", "facebook/nllb-200-distilled-600M")
SRC_LANG = "ben_Beng" 
TGT_LANG = "eng_Latn"
OUTPUT_DIR = "fine_tuned_model"
MAX_LENGTH = 128

# Create training data file if it doesn't exist
if not os.path.exists(TRAINING_DATA):
    print(f"Creating {TRAINING_DATA} with sample data...")
    sample_data = {
        "bangla": [
            "আসিফ",
            "আসিফ ইকবাল",
            "তোমার নাম কি আসিফ?",
            "আমার নাম আসিফ।",
            "সাদিয়া আক্তার",
            "মোহাম্মদ আসিফ ইকবাল",
        ],
        "english": [
            "Asif",
            "Asif Iqbal",
            "Is your name Asif?",
            "My name is Asif.",
            "Sadia Akter",
            "Mohammad Asif Iqbal",
        ]
    }
    pd.DataFrame(sample_data).to_csv(TRAINING_DATA, index=False)
    print(f"Created {TRAINING_DATA}. Add more examples as needed.")
else:
    print(f"Using existing {TRAINING_DATA}")

# Load training data
training_df = pd.read_csv(TRAINING_DATA)
print(f"Training with {len(training_df)} examples")

# Create dataset
dataset = Dataset.from_pandas(training_df)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Tokenization function
def tokenize_function(examples):
    # Process source
    source = examples["bangla"]
    target = examples["english"]
    
    # Tokenize source
    model_inputs = tokenizer(source, max_length=MAX_LENGTH, truncation=True)
    
    # Tokenize target
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target, max_length=MAX_LENGTH, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=dataset.column_names
)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=5,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest"
)

# Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

def main():
    print("Starting fine-tuning...")
    trainer.train()
    
    print("Saving fine-tuned model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Test the fine-tuned model
    print("\nTesting translations after fine-tuning:")
    test_inputs = [
        "আসিফ",
        "আমার নাম আসিফ ইকবাল",
        "সাদিয়া আক্তার"
    ]
    
    for text in test_inputs:
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
            max_new_tokens=128
        )
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"BN: {text}")
        print(f"EN: {translation}\n")
    
    print(f"Fine-tuned model saved to {OUTPUT_DIR}")
    print(f"To use it, set: $env:INDIGO_MODEL='{os.path.abspath(OUTPUT_DIR)}'")

if __name__ == "__main__":
    main()
