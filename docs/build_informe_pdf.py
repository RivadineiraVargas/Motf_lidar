"""
build_informe_pdf.py — Genera INFORME_FASE1.pdf desde INFORME_FASE1.md (markdown
+ figuras) usando markdown + fpdf2. Corre desde docs/.

    conda run -n sapiens_gpu python build_informe_pdf.py
"""
import os, re
import matplotlib
import markdown
from fpdf import FPDF

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, 'INFORME_FASE1.md')
OUT = os.path.join(HERE, 'INFORME_FASE1.pdf')
FONTDIR = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data/fonts/ttf')

text = open(MD, encoding='utf-8').read()
html = markdown.markdown(text, extensions=['tables'])

# rutas de imagen -> absolutas + ancho controlado para que no desborden
html = html.replace('src="figuras/', f'src="{HERE}/figuras/')
html = re.sub(r'<img ', '<img width="470" ', html)

pdf = FPDF()
# fuente Unicode (DejaVu) para acentos y em-dash en portugués
pdf.add_font('DejaVu', '', os.path.join(FONTDIR, 'DejaVuSans.ttf'))
pdf.add_font('DejaVu', 'B', os.path.join(FONTDIR, 'DejaVuSans-Bold.ttf'))
pdf.add_font('DejaVu', 'I', os.path.join(FONTDIR, 'DejaVuSans-Oblique.ttf'))
pdf.add_font('DejaVu', 'BI', os.path.join(FONTDIR, 'DejaVuSans-BoldOblique.ttf'))
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font('DejaVu', size=10)
pdf.write_html(html, font_family='DejaVu')
pdf.output(OUT)
print(f'PDF gerado: {OUT}')
