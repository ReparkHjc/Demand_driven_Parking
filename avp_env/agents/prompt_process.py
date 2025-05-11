
def build_parking_prompt(instruction: str, position: int, view: str) -> str:
    if view == 'combined':
        return f"""<image_placeholder>
        You are an autonomous parking assistant. Your job is to make a decision based on the following inputs:

        - The current position of the vehicle is {position}.
        - The image shows the current view from front/left/right/back side of the vehicle .
        - The parking instruction is: "{instruction}"

        You can choose one of the following actions:
        0: Move the vehicle one step forward and refresh the perception image.
        1: Attempt to park on the left side if there is a parking spot centrally aligned with the left side view of the vehicle.
        2: Attempt to park on the right side if there is a parking spot centrally aligned with the right side view of the vehicle.

        Based on the instruction and what you see in the image, what is the most appropriate action at this moment?

        Please respond with only a single number: 0, 1, or 2."""
    elif view == 'multi':
        return f"""
        You are an autonomous parking assistant. Your job is to make a decision based on the following inputs:

        - The current position of the vehicle is {position}.
        - The image shows the current view from front side of the vehicle: <image_placeholder><image>\n
        - The image shows the current view from left side of the vehicle: <image_placeholder><image>\n 
        - The image shows the current view from right side of the vehicle: <image_placeholder><image>\n
        - The image shows the current view from back side of the vehicle: <image_placeholder><image>\n
        - The parking instruction is: "{instruction}"

        You can choose one of the following actions:
        0: Move the vehicle one step forward and refresh the perception image.
        1: Attempt to park on the left side if there is a parking spot centrally aligned with the left side view of the vehicle.
        2: Attempt to park on the right side if there is a parking spot centrally aligned with the right side view of the vehicle.


        Based on the instruction and what you see in the image, what is the most appropriate action at this moment?

        Please respond with only a single number: 0, 1, or 2."""
    elif view == 'side':
        return f"""
        You are an autonomous parking assistant. Your job is to make a decision based on the following inputs:

        - The current position of the vehicle is {position}.
        - The image shows the current view from left side of the vehicle: <image_placeholder><image>\n 
        - The image shows the current view from right side of the vehicle: <image_placeholder><image>\n
        - The parking instruction is: "{instruction}"

        You can choose one of the following actions:
        0: Move the vehicle one step forward and refresh the perception image.
        1: Attempt to park on the left side if there is a parking spot centrally aligned with the left side view of the vehicle.
        2: Attempt to park on the right side if there is a parking spot centrally aligned with the right side view of the vehicle.


        Based on the instruction and what you see in the image, what is the most appropriate action at this moment?

        Please respond with only a single number: 0, 1, or 2."""
    elif view == 'right' or view == 'left' or view == 'back' or view == 'front':
        return f"""<image_placeholder>
        You are an autonomous parking assistant. Your job is to make a decision based on the following inputs:

        - The current position of the vehicle is {position}.
        - The image shows the current view from {view} side of the vehicle .
        - The parking instruction is: "{instruction}"

        You can choose one of the following actions:
        0: Move the vehicle one step forward and refresh the perception image.
        1: Attempt to park on the left side if there is a parking spot centrally aligned with the left side view of the vehicle.
        2: Attempt to park on the right side if there is a parking spot centrally aligned with the right side view of the vehicle.


        Based on the instruction and what you see in the image, what is the most appropriate action at this moment?

        Please respond with only a single number: 0, 1, or 2."""
    else:
        raise ValueError(
            f"Invalid view '{view}'. Choose from 'front', 'left', 'right', 'back', 'combined', 'multi', 'side'.")
