# DocProcessor

**Instructions:**
1. Install dependencies:
   ```bash
   pip install doctr[torch] layoutparser[ocr] deepdoctection[torch,tf,ocr] pdf2image opencv-python
   ```
   Also install Poppler (for pdf2image):  
   Ubuntu: `sudo apt-get install poppler-utils`  
   Mac: `brew install poppler`

2. Run the script (for pages 2–50, enable all options):
   ```bash
   python pdf_to_json_pipeline.py --pdf yourfile.pdf --output result.json --doctr --layoutparser --deepdoctection
   ```
- Each enabled tool's JSON results are saved under a separate key in the output file.
- You can enable/disable DocTR, LayoutParser, and Deepdoctection as desired by passing their respective flags.
