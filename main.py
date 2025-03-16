# main.py
import preprocess
import ocr
import os


def main():
    # Configuration
    input_image = "input/sample2.jpg"
    output_dir = "output"
    weights_path = "weights/pan_exp/weights/best.pt"

    # Step 1: Run Preprocessing
    print("Starting preprocessing...")
    preprocessed_image_path = preprocess.run_preprocess(input_image, output_dir)
    if preprocessed_image_path is None:
        print("Preprocessing failed. Exiting.")
        return

    # Step 2: Run OCR on the preprocessed image
    print("Starting OCR...")
    output_image_path = "output/annotated_sample2.jpg"
    output_json_path = "output/ocr_results_sample2.json"
    ocr.run_ocr(
        preprocessed_image_path, weights_path, output_image_path, output_json_path
    )

    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
