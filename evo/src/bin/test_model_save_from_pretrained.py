import torch
from peft import LoraConfig, TaskType
from model import EvoBinaryTokenClassificationModel
from transformers import AutoModelForCausalLM, AutoConfig
from peft.peft_model import PeftModel

def weight_modification_test_peft_adapter(output_dir="/work/dlclarge2/koeksalr-crispr/temp"):
    evo_config = AutoConfig.from_pretrained(
            "togethercomputer/evo-1-8k-base", trust_remote_code=True, revision="1.1_fix")
        
    # If this is set to True, it will not fit on the 80GB GPU
    evo_config.use_cache = False

    evo_model = AutoModelForCausalLM.from_pretrained(
          "togethercomputer/evo-1-8k-base",
          config=evo_config,
          trust_remote_code=True,
          revision="1.1_fix",
          device_map={"": 0},
          torch_dtype=torch.float16, 
          )
    
    # Apply LoRA configuration using hyperparameters
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=1,
        lora_alpha=32,
        target_modules=["l1"],  # Adjust based on your target module
        lora_dropout=0.0,
        bias="none"
    )
    peft_model = PeftModel(evo_model, peft_config=lora_config)
    original_model = EvoBinaryTokenClassificationModel(peft_model)

    # 1. Access PEFT adapter weights for verification
    peft_weights = original_model.backbone_model.base_model.model.backbone.blocks[0].mlp.l1.lora_A.default.weight

    # Clone original PEFT weights for comparison
    original_weights = peft_weights.clone().detach()

    # 2. Modify first weight in PEFT adapter (example modification)
    print("\nOriginal first weight value in PEFT adapter:", original_weights[0][0].item())
    peft_weights.data[0][0] = 10.0
    print("Modified first weight value in PEFT adapter:", peft_weights[0][0].item())

    # 3. Save modified model
    original_model.save_pretrained(output_dir)

    # 4. Reload modified model
    reloaded_model = EvoBinaryTokenClassificationModel.from_pretrained(output_dir)

    # 5. Verify weight persistence for modified elements in PEFT adapter
    reloaded_peft_weights = reloaded_model.backbone_model.base_model.model.backbone.blocks[0].mlp.l1.lora_A.default.weight
    
    assert reloaded_peft_weights[0][0].item() == 10.0, (
        "Modified PEFT adapter weight did not persist! "
        f"Expected 10.0, got {reloaded_peft_weights[0][0].item()}"
    )

    # 6. Verify other PEFT weights remain unchanged
    assert torch.allclose(reloaded_peft_weights, peft_weights), (
        "Other PEFT adapter weights changed unexpectedly!"
        # Print the weights
        f"\nOriginal weights:\n{peft_weights}\n"
        f"Reloaded weights:\n{reloaded_peft_weights}"
    )

    print("All tests passed for PEFT adapter!")

# Example usage
if __name__ == "__main__":
    weight_modification_test_peft_adapter()
