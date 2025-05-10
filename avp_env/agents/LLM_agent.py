import openai
import base64
import io

import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # 强制使用本地文件
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 禁用在线检查


class DSVL7BAgent:
    def __init__(self, model_path="../../deepseek-vl-7b-chat"):
        self.processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = self.model.to(torch.bfloat16).cuda().eval()

    def get_action(self, image: Image.Image, prompt: str) -> int:
        conversation = [
            {
                "role": "User",
                "content": prompt,
                "images": [image]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        prepare_inputs = self.processor(
            conversations=conversation,
            images=[image],
            force_batchify=True
        ).to(self.model.device)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=10,
            do_sample=False,
            use_cache=True
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()

        try:
            action = int(answer)
            if action in [0, 1, 2]:
                return action
        except:
            pass

        return 0  # 默认返回前进


class multiImgMultimodalLLMAgent:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.model = "gpt-4-vision-preview"

    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode()

    def get_action(self, images: list, instruction: str, position: int) -> int:
        """
        images: [front, left, right, rear] as PIL Images
        instruction: parking instruction (str)
        position: current position (int)
        """
        assert len(images) == 4, "Expected 4 images: front, left, right, rear"

        image_inputs = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(img)}"}}
            for img in images
        ]

        prompt = {
            "type": "text",
            "text": (
                f"You are an autonomous parking assistant.\n"
                f"The vehicle is currently at position {position}.\n"
                f"You are given four camera images from the vehicle:\n"
                f"- Image 1: Front view\n"
                f"- Image 2: Left view\n"
                f"- Image 3: Right view\n"
                f"- Image 4: Rear view\n\n"
                f"Parking instruction: \"{instruction}\"\n\n"
                f"Choose one action:\n"
                f"0: Move forward\n"
                f"1: Attempt to park on the left\n"
                f"2: Attempt to park on the right\n\n"
                f"Please respond with only a single number: 0, 1, or 2."
            )
        }

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": [prompt] + image_inputs}],
            max_tokens=10,
            temperature=0.2,
        )

        reply = response.choices[0].message.content.strip()
        try:
            action = int(reply)
            if action in [0, 1, 2]:
                return action
        except ValueError:
            print("⚠️ Could not parse action from LLM output:", reply)

        return 0  # default fallback
