import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline

class ESGModel:
    def __init__(self, model_path):
        # Force everything to run on CPU
        self.device = torch.device("cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = self._create_pipeline()

    def _create_pipeline(self):
        from transformers import pipeline

        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,  # Force the pipeline to run on CPU
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=True,
        )

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_pipeline(self):
        return HuggingFacePipeline(pipeline=self.pipeline)
