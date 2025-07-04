import argparse
import json
import numpy as np

from pathlib import Path
from pdf2image import convert_from_path

# DocTR OCR
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# LayoutParser
import layoutparser as lp

# Deepdoctection
from deepdoctection import DoctectionPipe
from deepdoctection.extern.pdf import poppler_to_image

def pdf_to_images(pdf_path):
    """Convert all PDF pages to images using pdf2image."""
    images = convert_from_path(pdf_path)
    return images

def run_doctr_ocr(images):
    """Run DocTR OCR on a list of images."""
    model = ocr_predictor(pretrained=True)
    results = []
    for img in images:
        result = model(img)
        results.append(result.export())
    return results

def run_layoutparser(images, model_name="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"):
    """Run LayoutParser on a list of images."""
    model = lp.Detectron2LayoutModel(model_name, extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5])
    layouts = []
    for img in images:
        np_img = np.array(img)
        layout = model.detect(np_img)
        layouts.append(layout.to_dict())
    return layouts

def run_deepdoctection(pdf_path, config="default"):
    """Run Deepdoctection on the PDF directly."""
    pipe = DoctectionPipe.from_config(config)
    images = list(poppler_to_image(pdf_path))
    results = []
    for img in images:
        doc = pipe.analyze(img)
        results.append(doc.as_json())
    return results

def main(pdf_path, output_json, enable_doctr, enable_layoutparser, enable_deepdoctection):
    images = pdf_to_images(pdf_path)
    results = {}

    if enable_doctr:
        print("Running DocTR OCR...")
        results['doctr'] = run_doctr_ocr(images)

    if enable_layoutparser:
        print("Running LayoutParser...")
        results['layoutparser'] = run_layoutparser(images)

    if enable_deepdoctection:
        print("Running Deepdoctection...")
        results['deepdoctection'] = run_deepdoctection(pdf_path)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Extraction complete. Results saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF to JSON pipeline using DocTR, LayoutParser, Deepdoctection")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--doctr", action="store_true", help="Enable DocTR OCR")
    parser.add_argument("--layoutparser", action="store_true", help="Enable LayoutParser")
    parser.add_argument("--deepdoctection", action="store_true", help="Enable Deepdoctection")
    
    args = parser.parse_args()

    main(
        pdf_path=args.pdf,
        output_json=args.output,
        enable_doctr=args.doctr,
        enable_layoutparser=args.layoutparser,
        enable_deepdoctection=args.deepdoctection,
    )