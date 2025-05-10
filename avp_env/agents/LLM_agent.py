import openai
import base64
import io
import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM,DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl.utils.io import load_pil_images
from qwen_vl_utils import process_vision_info
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

os.environ["HF_HUB_OFFLINE"] = "1"  # 强制使用本地文件
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 禁用在线检查

class JanusAgent:
    def __init__(self, model_path="../../Janus-Pro-7B"):
        self.processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        self.model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = self.model.to(torch.bfloat16).cuda().eval()

    def get_action(self, image: Image.Image, prompt: str) -> int:
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)

        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
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


class DeepseekVL2Agent:
    def __init__(self, model_path="../../deepseek-vl2-tiny"):
        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        self.model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = self.model.to(torch.bfloat16).cuda().eval()

    def get_action(self, image: Image.Image, prompt: str) -> int:
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\n" + prompt,
                "images": [image]
            },
            {
                "role": "<|Assistant|>",
                "content": ""
            }
        ]

        pil_images = load_pil_images(conversation)

        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
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


class QwenVLAgent:
    def __init__(self, model_path="../../Qwen2.5-VL-7B-Instruct"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def get_action(self, image: Image.Image, prompt: str) -> int:
        # 构建符合Qwen格式的message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 构建文本模板
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 视觉信息预处理
        image_inputs, video_inputs = process_vision_info(messages)

        # 构建最终输入
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # 推理生成
        generated_ids = self.model.generate(**inputs, max_new_tokens=10)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        # 解析整数动作
        try:
            action = int(output_text)
            if action in [0, 1, 2]:
                return action
        except:
            pass

        return 0  # 默认返回前进
