from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
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
    def __init__(
        self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 64, temperature: float = 0.07
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        # add projection layers for vision and text encoders
        # try to infer hidden dims from encoders
        try:
            vision_dim = getattr(self.vision_encoder.config, "hidden_size", None)
        except Exception:
            vision_dim = None
        try:
            text_dim = getattr(self.text_encoder.config, "hidden_size", None)
        except Exception:
            text_dim = None

        # Fallback if not found
        if vision_dim is None:
            # default to 768
            vision_dim = 768
        if text_dim is None:
            text_dim = 768

        self.vision_proj = nn.Linear(vision_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)

        # Learnable logit scale (as in OpenAI/CLIP)
        # align dtype of projection layers and logit scale to vision encoder dtype if available
        try:
            param_dtype = next(self.vision_encoder.parameters()).dtype
        except StopIteration:
            param_dtype = torch.float32

        logit_value = torch.tensor(1.0 / temperature).log()
        try:
            logit_value = logit_value.to(param_dtype)
        except Exception:
            pass
        self.logit_scale = nn.Parameter(logit_value)
        # convert projection layers to encoder dtype to avoid dtype mismatch during forward
        self.vision_proj = self.vision_proj.to(dtype=param_dtype)
        self.text_proj = self.text_proj.to(dtype=param_dtype)

        # Ensure these layers are trainable
        # NOTE: get_target_modules_for_lora excludes 'projection', but these projection layers should be trained
        # via PEFT/LoRA or normally as additional weights.
        self.proj_dim = proj_dim
        self.temperature = temperature

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_encoder(pixel_values=image, return_dict=True)
        # Try common pooling strategies
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            img_emb = outputs.pooler_output
        else:
            # fallback: take the first token (CLS) if present or mean pool over sequence
            last_hidden = outputs.last_hidden_state
            img_emb = last_hidden[:, 0, :] if last_hidden.shape[1] > 1 else last_hidden.mean(dim=1)
        return img_emb

    def encode_text(self, text: str) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=text["input_ids"], attention_mask=text.get("attention_mask", None), return_dict=True)
        # pooling: use last non-pad token
        last_hidden = outputs.last_hidden_state
        if text.get("attention_mask", None) is not None:
            mask = text["attention_mask"].long()
            lengths = mask.sum(dim=1) - 1
            batch_idx = torch.arange(last_hidden.shape[0], device=last_hidden.device)
            txt_emb = last_hidden[batch_idx, lengths]
        else:
            txt_emb = last_hidden[:, -1, :]

        return txt_emb

    def save_pretrained(self, save_directory: str, **kwargs):
        """Customize save method, save additional parameters"""

        additional_state_dict = {}
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            additional_state_dict[name] = param.data

        torch.save(additional_state_dict, Path(save_directory) / "additional_weights.pt")

    def load_pretrained(self, load_directory: str, **kwargs):
        """Customize load method, load projection additional parameters"""

        additional_weights_path = Path(load_directory) / "additional_weights.pt"
        if additional_weights_path.exists():
            additional_state_dict = torch.load(additional_weights_path, map_location="cpu")

            for name, param in self.named_parameters():
                if "vision_encoder." in name or "text_encoder." in name:
                    continue
                param.data = additional_state_dict[name]

    def set_trainable_parameters(self):
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            param.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Enable gradient checkpointing for the vision and text backbones.
        (You don't need to touch this method)
        """
        self.vision_encoder.gradient_checkpointing_enable(**kwargs)
        self.text_encoder.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        """
        Enable input require grads for the vision and text backbones.
        (You don't need to touch this method)
        """

        # Reference: https://discuss.huggingface.co/t/peft-lora-gpt-neox-backward-pass-failing/35641
        def make_inputs_require_grads(module, input, output):  # noqa: A002
            output.requires_grad_(True)

        self.vision_encoder.embeddings.register_forward_hook(make_inputs_require_grads)
        self.text_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CLIP model.
        Args:
            pixel_values: The pixel values of the image.
            input_ids: The input ids of the text.
            attention_mask: The attention mask of the text.
            labels: The labels for the text features.
            (NOTE: you don't need to use the variable `labels`, this is just for compatibility with the Trainer class)
            (Hint: refer to returned values of the __getitem__ method in the CaptionDatasetForTraining class)
        Returns:
            TODO: think about the what values should be returned
        """
        # pixel_values: (B_img, C, H, W)
        # input_ids: (B_txt, seq_len)
        # attention_mask: (B_txt, seq_len)

        # support being called with a single dict positional (PEFT Trainer may pass inputs as args)
        if len(args) == 1 and isinstance(args[0], dict):
            inputs = args[0]
            pixel_values = inputs.get("pixel_values", pixel_values)
            input_ids = inputs.get("input_ids", input_ids)
            attention_mask = inputs.get("attention_mask", attention_mask)
            labels = inputs.get("labels", labels)
        elif len(args) >= 1 and isinstance(args[0], torch.Tensor):
            # support positional tensor arguments (pixel_values, input_ids, attention_mask)
            pixel_values = pixel_values or args[0]
            if len(args) > 1 and isinstance(args[1], torch.Tensor):
                input_ids = input_ids or args[1]
            if len(args) > 2 and isinstance(args[2], torch.Tensor):
                attention_mask = attention_mask or args[2]

        # ensure pixel_values dtype matches encoder weights (e.g., bfloat16)
        try:
            enc_dtype = next(self.vision_encoder.parameters()).dtype
            pixel_values = pixel_values.to(dtype=enc_dtype)
        except StopIteration:
            pass

        # compute image embedding
        vision_out = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        if hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
            image_emb = vision_out.pooler_output
        else:
            last_hidden = vision_out.last_hidden_state
            image_emb = last_hidden[:, 0, :] if last_hidden.shape[1] > 1 else last_hidden.mean(dim=1)

        # project and normalize
        vision_feature = self.vision_proj(image_emb)
        vision_feature = F.normalize(vision_feature, dim=-1)

        # compute text embeddings
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = text_out.last_hidden_state
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(last_hidden.shape[0], device=last_hidden.device)
            text_emb = last_hidden[batch_idx, lengths]
        else:
            text_emb = last_hidden[:, -1, :]

        text_feature = self.text_proj(text_emb)
        text_feature = F.normalize(text_feature, dim=-1)

        # compute logits using learnable logit scale
        logit_scale = self.logit_scale.exp()
        # Matmul: (B_img, D) @ (B_txt, D).T -> (B_img, B_txt)
        logits = logit_scale * torch.matmul(vision_feature, text_feature.T)

        return vision_feature, text_feature, logits


def compute_clip_loss(model: nn.Module, inputs: dict[str, torch.Tensor], return_outputs: bool = False):
    """
    Compute the loss for the CLIP model.
    Args:
        outputs: A tuple containing the outputs of CLIP.forward().
        labels: The labels for the text features.
        (NOTE: you don't need to use the variable `labels`, this is just for compatibility with the Trainer class)
        num_items_in_batch: The number of items in the batch.
        (NOTE: you don't need to use the variable `num_items_in_batch`, this is just for compatibility with Trainer)
    Returns:
        The loss for the CLIP model.
    """
    # This compute_loss signature matches transformers.Trainer.compute_loss(model, inputs)
    # inputs will be a dict containing pixel_values, input_ids, attention_mask, labels (labels unused)
    device = next(model.parameters()).device
    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Forward pass via kwargs (works better with PeftModel/Trainer wrappers)
    outputs_tuple = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
    if isinstance(outputs_tuple, tuple) or isinstance(outputs_tuple, list):
        vision_feature, text_feature, logits = outputs_tuple[:3]
    elif isinstance(outputs_tuple, dict):
        vision_feature = outputs_tuple.get("vision_feature")
        text_feature = outputs_tuple.get("text_feature")
        logits = outputs_tuple.get("logits")
    else:
        # fallback: try using it as a tuple
        vision_feature, text_feature, logits = outputs_tuple

    # We assume batch sizes match for image and text for training
    b_img = vision_feature.shape[0]
    b_txt = text_feature.shape[0]
    min_b = min(b_img, b_txt)
    labels = torch.arange(min_b, device=device)

    loss_fct = nn.CrossEntropyLoss()
    # compute loss on square submatrix if shapes mismatch (e.g., 1 x N eval cases)
    logits_sub = logits[:min_b, :min_b]
    loss_i2t = loss_fct(logits_sub, labels)
    loss_t2i = loss_fct(logits_sub.T, labels)
    loss = (loss_i2t + loss_t2i) / 2.0

    if return_outputs:
        return loss, (vision_feature, text_feature, logits)
    return loss


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
    train_dataset_name: str = "train",
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
    model.set_trainable_parameters()

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
    train_dataset = CaptionDataset(train_dataset_name, data_dir)
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

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
            # Accept num_items_in_batch for compatibility with some Trainer versions
            return compute_clip_loss(model, inputs, return_outputs=return_outputs)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
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

    Fire({"train": train, "test": test, "demo_train": demo_train})


if __name__ == "__main__":
    main()
