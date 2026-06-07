import markdown
from xhtml2pdf import pisa
import os

md_file = "TFM_WMS_Desarrollo.md"
pdf_file = "TFM_WMS_Desarrollo.pdf"

# Read markdown content
with open(md_file, "r", encoding="utf-8") as f:
    text = f.read()

# Convert markdown to html with some basic styling
html_content = markdown.markdown(text, extensions=['tables'])

# Add CSS for better formatting
styled_html = f"""
<html>
<head>
<style>
    @page {{
        size: letter portrait;
        margin: 2cm;
    }}
    body {{
        font-family: Arial, sans-serif;
        font-size: 12pt;
        line-height: 1.5;
        color: #333;
    }}
    h1 {{ color: #004080; font-size: 20pt; border-bottom: 2px solid #004080; padding-bottom: 5px; }}
    h2 {{ color: #0059b3; font-size: 16pt; margin-top: 20px; }}
    h3 {{ color: #0073e6; font-size: 14pt; }}
    p {{ text-align: justify; margin-bottom: 15px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; color: #333; font-weight: bold; }}
    code {{ background-color: #f8f8f8; padding: 2px 4px; border-radius: 4px; font-family: monospace; font-size: 10pt; }}
</style>
</head>
<body>
    {html_content}
</body>
</html>
"""

# Convert HTML to PDF
with open(pdf_file, "w+b") as result_file:
    pisa_status = pisa.CreatePDF(styled_html, dest=result_file)

if pisa_status.err:
    print("Error during PDF conversion")
else:
    print("PDF created successfully!")
