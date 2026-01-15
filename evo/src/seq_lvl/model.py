import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from peft import PeftConfig, get_peft_model
from peft.peft_model import PeftModel

class EvoMultiClassSequenceClassificationModel(nn.Module):
    def __init__(self, config: PeftConfig,
                 pretrained_model_dir=None,
                 dropout_prob=0.0,
                 num_classes=3):
        super().__init__()
        evo_config = AutoConfig.from_pretrained(
            "togethercomputer/evo-1-8k-base", trust_remote_code=True, revision="1.1_fix")
        
        # Disable cache to fit large models
        evo_config.use_cache = False

        evo_model = AutoModelForCausalLM.from_pretrained(
              "togethercomputer/evo-1-8k-base",
              config=evo_config,
              trust_remote_code=True,
              revision="1.1_fix",
              device_map={"": 0},
              torch_dtype=torch.float16, 
              )
        
        backbone_model = None
        if pretrained_model_dir is not None:
            backbone_model = PeftModel.from_pretrained(evo_model,
                                                       pretrained_model_dir,
                                                       config=config)
        else:
            backbone_model = get_peft_model(evo_model, config)
            
        self.backbone_model = backbone_model
        self.dropout = nn.Dropout(dropout_prob)
        
        # Adjust classifier for sequence-level output
        self.classifier = nn.Linear(512, num_classes)  # Output size matches number of classes

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through base model
        outputs = self.backbone_model(input_ids, attention_mask=attention_mask)
        
        # Pooling: Use mean pooling over token embeddings
        pooled_output = torch.mean(outputs.logits, dim=1)  # Shape: (batch_size, hidden_dim)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output.float())  # Shape: (batch_size, num_classes)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Compute loss at the sequence level
            loss = loss_fct(logits.view(-1, 3), labels.view(-1).long())

        return {"loss": loss, "logits": logits}

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.backbone_model.save_pretrained(save_directory)

        dropout_path = os.path.join(save_directory, "dropout_prob.txt")
        with open(dropout_path, "w") as f:
            f.write(str(self.dropout.p))
        
        classifier_path = os.path.join(save_directory, "classifier.pt")
        torch.save(self.classifier.state_dict(), classifier_path)

    @classmethod
    def from_pretrained(cls, model_dir):
        config = PeftConfig.from_pretrained(model_dir)
        
        dropout_path = os.path.join(model_dir, "dropout_prob.txt")
        with open(dropout_path, "r") as f:
            dropout_prob = float(f.read())

        model = cls(config, model_dir, dropout_prob=dropout_prob)

        classifier_path = os.path.join(model_dir, "classifier.pt")
        model.classifier.load_state_dict(torch.load(classifier_path))
        
        return model

    