from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import openai
import base64
import io

class combineMultimodalLLMAgent:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.model.eval()

    def get_action(self, image: Image.Image, instruction: str, position: int) -> int:
        prompt = f"""
        You are an autonomous parking assistant. Your job is to make a decision based on the following inputs:
        
        - The current position of the vehicle is {position}.
        - The image shows the current view from the vehicle.
        - The parking instruction is: "{instruction}"
        
        You can choose one of the following actions:
        0: Move forward.
        1: Attempt to park on the left side.
        2: Attempt to park on the right side.
        
        Based on the instruction and what you see in the image, what is the most appropriate action at this moment?
        
        Please respond with only a single number: 0, 1, or 2.
        """

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=10)
        result = self.processor.batch_decode(output, skip_special_tokens=True)[0].strip()

        try:
            action = int(result)
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
