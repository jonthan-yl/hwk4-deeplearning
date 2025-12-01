from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load(model_name: str = "clip_model"):
    from pathlib import Path

    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM()
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    clip = CLIP(vision_encoder, text_encoder)
    clip = PeftModel.from_pretrained(clip, model_path).to(device)

    clip.model.load_pretrained(model_path)
    clip.model.eval()
    if device == "cuda":
        clip = clip.to(dtype=torch.bfloat16)

    return clip


def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Custom data collator for CLIP training.
    """
    # Get max sequence length
    max_length = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])  # assume all are same shape
    labels = torch.stack([pad_tensor(f["labels"], pad_value=-100) for f in features])

    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "pixel_values": pixel_values.float(),
        "labels": labels.long(),
    }


class CaptionDatasetForTraining(Dataset):
    def __init__(self, dataset: CaptionDataset, processor: AutoProcessor):
        self.dataset = dataset
        self.image_processor = tv.transforms.Compose(
            [
                tv.transforms.Resize(192),
                tv.transforms.RandomResizedCrop(192, scale=(0.5, 1.0)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.image_processor(image)
        text = item["caption"] + self.processor.tokenizer.eos_token
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        input_ids = text_inputs["input_ids"].squeeze(0).long()
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,  # placeholder to fit the collator
        }


class CLIP(nn.Module):
    def __init__(self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 64, temperature: float = 0.07):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.proj_dim = proj_dim
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Learnable linear projections for features
        self.image_proj = nn.Linear(vision_encoder.config.hidden_size, proj_dim)
        self.text_proj = nn.Linear(text_encoder.config.hidden_size, proj_dim)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.vision_encoder(image)[0]  # usually last_hidden_state or pooled
        return self.image_proj(feats.mean(dim=1))  # pool if needed

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        feats = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        return self.text_proj(feats[:, 0, :])  # use [CLS] token

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, labels: torch.Tensor = None):
        image_feats = self.encode_image(pixel_values)
        text_feats = self.encode_text(input_ids, attention_mask)

        # Normalize
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # similarity
        logits = torch.matmul(image_feats, text_feats.T) / self.temperature
        return image_feats, text_feats, logits



def compute_clip_loss(outputs, labels=None, num_items_in_batch=None):
    image_feats, text_feats, logits = outputs
    batch_size = image_feats.shape[0]
    target = torch.arange(batch_size, device=image_feats.device)
    loss_i2t = nn.CrossEntropyLoss()(logits, target)
    loss_t2i = nn.CrossEntropyLoss()(logits.T, target)
    return (loss_i2t + loss_t2i) / 2

def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    target_modules = []
    for name, module in model.named_modules():
        # if isinstance(module, nn.Linear) and ("vision_encoder" in name and "projection" not in name):
        if (
            isinstance(module, nn.Linear)
            and ("vision_encoder" in name or "text_encoder" in name)
            and "projection" not in name
        ):
            target_modules.append(name)

    return target_modules


def train(
    data_dir: Path | None = None,
    output_dir: str = "clip",
    num_train_epochs: float = 0.2,  # for debugging purpose, increase this once the dry run works
    per_device_train_batch_size: int = 128,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-4,
    num_workers: int = 16,
):
    vlm = BaseVLM()

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize model and processor
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    model = CLIP(vision_encoder, text_encoder).to(device).bfloat16()

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        # target_modules="all-linear",
        target_modules=get_target_modules_for_lora(model),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # load dataset
    train_dataset = CaptionDataset("train", data_dir)
    train_dataset = CaptionDatasetForTraining(train_dataset, processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        bf16=True if device == "cuda" else False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        label_names=["labels"],
        dataloader_num_workers=num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
        compute_loss_func=compute_clip_loss,
    )

    trainer.train()

    # save model
    trainer.save_model(output_dir)
    model.model.save_pretrained(output_dir)

    writer.close()

    return model, processor


def demo_train():
    train(
        train_dataset_name="train_demo",
        output_dir="demo_clip",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        num_workers=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-8,
    )


def test(ckpt_path: str, val_dataset: str = "valid_grader"):
    import tqdm

    testset = MultiChoiceQADataset(val_dataset)

    clip = load(ckpt_path)
    clip = clip.model.to(device)

    image_processor = tv.transforms.Compose(
        [
            tv.transforms.Resize(192),
            tv.transforms.CenterCrop(192),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    correct_count = 0
    total_count = 0

    for pair in tqdm.tqdm(testset):
        image = Image.open(pair["image_path"]).convert("RGB")
        pixel_values = image_processor(image).unsqueeze(0).to(device).bfloat16()
        text_inputs = processor(
            text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"].long().to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        vision_feature, text_feature, _ = clip(pixel_values, input_ids, attention_mask)
        prediction = torch.matmul(vision_feature, text_feature.T).argmax(dim=-1)
        if prediction == pair["correct_index"]:
            correct_count += 1
        total_count += 1

    print(f"Accuracy: {correct_count / total_count}")


def main():
    from fire import Fire

    Fire({"train": train, "test": test})


if __name__ == "__main__":
    main()
