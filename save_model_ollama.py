from unsloth import FastLanguageModel
import os

# Load your fine-tuned model and tokenizer
huggingface_model_name = "AchrafGhribi31/llama3-fine-tuned_ESG-QA_4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=huggingface_model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,  # Ensure this is set to True for 4-bit quantization
)

# Prepare the model for inference (Trion not required)
FastLanguageModel.for_inference(model)

# Save the model to GGUF format and push to Hugging Face
model.push_to_hub_gguf(
    "AchrafGhribi31/llama3-fine-tuned_ESG-QA_4bit-gguf",  # Adjust to your username and model name
    tokenizer,
    quantization_method=["q4_k_m"],  # Use 4-bit quantization method
    token=os.getenv("HF_TOKEN"),
)

print("Model saved and pushed to Hugging Face in GGUF format with 4-bit quantization.")
