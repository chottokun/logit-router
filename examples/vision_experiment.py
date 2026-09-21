import argparse
import json

from PIL import Image, ImageDraw

from logit_router.vision_router import VisionLogitRouter


def create_test_image(type_name: str) -> Image.Image:
    img = Image.new("RGB", (224, 224), color="white")
    draw = ImageDraw.Draw(img)

    if type_name == "color_red":
        draw.rectangle([50, 50, 174, 174], fill="red")
    elif type_name == "color_blue":
        draw.rectangle([50, 50, 174, 174], fill="blue")
    elif type_name == "shape_circle":
        draw.ellipse([50, 50, 174, 174], fill="green")
    elif type_name == "ui_error":
        draw.rectangle([10, 10, 214, 214], fill="lightgray")
        draw.rectangle([20, 20, 204, 60], fill="red")
        # Text is tricky to draw without font, so just simulate error box
    return img


def run_experiment(
    router: VisionLogitRouter,
    image_type: str,
    context: str,
    instruction: str,
    choices: list[str],
):
    print(f"\n--- Running Experiment: {image_type} ---")
    img = create_test_image(image_type)

    print(f"Context: {context}")
    print(f"Instruction: {instruction}")
    print(f"Choices: {choices}")

    result = router.route(
        image=img, context=context, instruction=instruction, choices=choices
    )

    if hasattr(result, "__dict__"):
        res_dict = result.__dict__
    else:
        res_dict = result

    print(json.dumps(res_dict, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Vision Experiment Script")
    parser.add_argument("--model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"Initializing VisionLogitRouter ({args.model} on {args.device}) ...")
    router = VisionLogitRouter(model_id=args.model, device=args.device)

    # Test Case 1: Color
    run_experiment(
        router,
        "color_red",
        "An image with a geometric shape.",
        "What color is the shape in the center of the image?",
        ["Red", "Green", "Blue", "Yellow"],
    )

    # Test Case 2: Shape
    run_experiment(
        router,
        "shape_circle",
        "An image with a geometric shape.",
        "What shape is in the center of the image?",
        ["Square", "Circle", "Triangle"],
    )

    # Test Case 3: UI Error Triage
    run_experiment(
        router,
        "ui_error",
        "A screenshot of a web application showing a notification banner.",
        "Based on the visual indicators (red banner), what is the most likely status of this notification?",
        ["Success", "Warning", "Error", "Info"],
    )


if __name__ == "__main__":
    main()
