#!/usr/bin/env python3
"""Draw the blog's TLR contraction diagrams as editable, accessible SVGs.

All three stages use qm=2, qk=3, qn=3, tile extent=256 and rank=64.
Block sizes in the diagrams are schematic; numerical dimensions are explicit.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/images/gsoc-2026'
OUT.mkdir(parents=True, exist_ok=True)
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
COLORS = {
    'blue': ('#dbeafe', '#60a5fa'),
    'orange': ('#ffedd5', '#fb923c'),
    'green': ('#dcfce7', '#4ade80'),
    'purple': ('#ede9fe', '#a78bfa'),
    'teal': ('#ccfbf1', '#2dd4bf'),
    'gold': ('#fef3c7', '#fbbf24'),
}
INK = '#172b42'
MUTED = '#42556b'


class Diagram:
    def __init__(self, name, title, subtitle, description, height=650):
        self.name = name
        self.root = ET.Element(f'{{{NS}}}svg', {
            'width': '1200', 'height': str(height),
            'viewBox': f'0 0 1200 {height}', 'role': 'img',
            'aria-labelledby': 'diagram-title diagram-description',
        })
        self.el('title', {'id': 'diagram-title'}, title)
        self.el('desc', {'id': 'diagram-description'}, description)
        self.rect(1, 1, 1198, height - 2, '#ffffff', '#dce4ed', radius=18)
        self.text(48, 55, title, 32, weight='bold')
        self.text(48, 98, subtitle, 23, color=MUTED)

    def el(self, tag, attrs, text=None):
        element = ET.SubElement(self.root, f'{{{NS}}}{tag}', {key: str(value) for key, value in attrs.items()})
        if text is not None:
            element.text = text
        return element

    def rect(self, x, y, w, h, fill, stroke='#dce4ed', radius=0, sw=2):
        self.el('rect', {'x': x, 'y': y, 'width': w, 'height': h, 'rx': radius,
                         'fill': fill, 'stroke': stroke, 'stroke-width': sw})

    def text(self, x, y, value, size=22, anchor='start', color=INK, weight='normal'):
        self.el('text', {'x': x, 'y': y, 'fill': color,
                         'font-family': 'DejaVu Sans, Arial, sans-serif', 'font-size': size,
                         'font-weight': weight, 'text-anchor': anchor}, value)

    def line(self, x1, y1, x2, y2, color='#93a3b5', sw=2):
        self.el('line', {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                         'stroke': color, 'stroke-width': sw})

    def arrow(self, x1, y1, x2, y2, color='#0f766e'):
        self.line(x1, y1, x2, y2, color, 3)
        self.el('polygon', {'points': f'{x2},{y2} {x2-12},{y2-7} {x2-12},{y2+7}', 'fill': color})

    def matrix(self, x, y, w, h, rows, cols, family, labels, heading, dims,
               highlight=None, row_labels=False, col_labels=False):
        fill, stroke = COLORS[family]
        self.text(x + w/2, y - 49, heading, 24, 'middle', weight='bold')
        self.text(x + w/2, y - 17, dims, 23, 'middle', color=MUTED)
        self.rect(x, y, w, h, fill, stroke, sw=2.5)
        for row in range(1, rows):
            self.line(x, y + h*row/rows, x + w, y + h*row/rows, stroke)
        for col in range(1, cols):
            self.line(x + w*col/cols, y, x + w*col/cols, y + h, stroke)
        if highlight is not None:
            row, col = highlight
            self.rect(x + w*col/cols + 2, y + h*row/rows + 2,
                      w/cols - 4, h/rows - 4, '#fef3c7', '#b45309', sw=3)
        for row in range(rows):
            for col in range(cols):
                self.text(x + w*(col + 0.5)/cols, y + h*(row + 0.5)/rows + 8,
                          labels[row][col], 23, 'middle')
        if row_labels:
            for row in range(rows):
                self.text(x - 13, y + h*(row + 0.5)/rows + 6, f'k={row+1}', 18, 'end', color=MUTED)
        if col_labels:
            for col in range(cols):
                self.text(x + w*(col + 0.5)/cols, y + h + 27, f'j={col+1}', 18, 'middle', color=MUTED)

    def metrics(self, cards, y=552):
        gap = 18
        width = (1104 - gap*(len(cards)-1))/len(cards)
        for idx, (top, bottom) in enumerate(cards):
            x = 48 + idx*(width + gap)
            self.rect(x, y, width, 66, '#f4f7fb', radius=10, sw=0)
            top_size, bottom_size = 23, 19
            for value, bold, size in [(top, True, top_size), (bottom, False, bottom_size)]:
                suffix = '-Bold' if bold else ''
                font_path = f'/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf'
                while ImageFont.truetype(font_path, size).getlength(value) > width - 28:
                    size -= 1
                if bold:
                    top_size = size
                else:
                    bottom_size = size
            self.text(x + width/2, y + 27, top, top_size, 'middle', weight='bold')
            self.text(x + width/2, y + 53, bottom, bottom_size, 'middle', color=MUTED)
        self.text(48, 640, 'Schematic blocks; numerical matrix dimensions are shown above each panel.', 17, color=MUTED)

    def save(self):
        ET.indent(self.root)
        path = OUT / f'{self.name}.svg'
        ET.ElementTree(self.root).write(path, encoding='utf-8', xml_declaration=True)
        print(path)
        self.render_png(path.with_suffix('.png'))

    def render_png(self, path):
        """Render our SVG primitive subset with Unicode-aware font support."""
        scale = 2
        height = int(self.root.attrib['height'])
        image = Image.new('RGB', (1200*scale, height*scale), '#ffffff')
        draw = ImageDraw.Draw(image)
        for element in self.root:
            tag = element.tag.rsplit('}', 1)[-1]
            a = element.attrib
            if tag == 'rect':
                x, y, w, h = [float(a[key])*scale for key in ['x', 'y', 'width', 'height']]
                draw.rounded_rectangle((x, y, x+w, y+h),
                                       radius=float(a.get('rx', 0))*scale,
                                       fill=a['fill'], outline=a['stroke'],
                                       width=round(float(a['stroke-width'])*scale))
            elif tag == 'line':
                coords = [float(a[key])*scale for key in ['x1', 'y1', 'x2', 'y2']]
                draw.line(coords, fill=a['stroke'], width=round(float(a['stroke-width'])*scale))
            elif tag == 'polygon':
                points = [tuple(float(v)*scale for v in pair.split(',')) for pair in a['points'].split()]
                draw.polygon(points, fill=a['fill'])
            elif tag == 'text':
                suffix = '-Bold' if a.get('font-weight') == 'bold' else ''
                font = ImageFont.truetype(f'/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf',
                                         round(float(a['font-size'])*scale))
                anchor = {'start': 'ls', 'middle': 'ms', 'end': 'rs'}[a['text-anchor']]
                draw.text((float(a['x'])*scale, float(a['y'])*scale), element.text,
                          font=font, fill=a['fill'], anchor=anchor)
        image.save(path)
        print(path)


def stage1():
    d = Diagram('tlr_stage1_panels', 'Stage 1 · Fuse the coupling panels',
                'Fix k. Combine two output tile rows and three output tile columns.',
                'A 128 by 256 left panel multiplies a 256 by 192 right panel, forming six 64 by 64 coupling blocks in one GEMM. Repeat independently for three k values.')
    d.text(600, 145, 'Sₖ = [V₁ₖᵀ; V₂ₖᵀ] [Wₖ₁  Wₖ₂  Wₖ₃]', 27, 'middle')
    d.matrix(60, 302, 300, 150, 2, 1, 'blue', [['V₁ₖᵀ'], ['V₂ₖᵀ']], 'Left coupling panel', '128 × 256')
    d.text(401, 386, '×', 38, 'middle')
    d.matrix(450, 227, 225, 300, 1, 3, 'orange', [['Wₖ₁', 'Wₖ₂', 'Wₖ₃']], 'Right coupling panel', '256 × 192')
    d.text(718, 386, '=', 38, 'middle')
    d.matrix(765, 302, 225, 150, 2, 3, 'green',
             [['S₁ₖ₁', 'S₁ₖ₂', 'S₁ₖ₃'], ['S₂ₖ₁', 'S₂ₖ₂', 'S₂ₖ₃']], 'Coupling block Sₖ', '128 × 192')
    d.metrics([('6 small products → 1 GEMM', 'for each contraction tile k'),
               ('(M, N, K) = (128, 192, 256)', 'free modes: (a, i) and (b, j)'),
               ('3 independent GEMMs', 'one grouped task per k')])
    d.save()


def stage2():
    d = Diagram('tlr_stage2_terminal', 'Stage 2 · Write the terminal workspace',
                'One task: i=1, k=2, j=1. All 18 tasks write directly into T₁ or T₂.',
                'A 64 by 64 coupling block multiplies a 64 by 256 transposed inner factor. Its 64 by 256 result is written into the k=2, j=1 block of the 192 by 768 terminal matrix T1.')
    d.text(600, 145, 'Tᵢₖⱼ = Sᵢₖⱼ Zₖⱼᵀ', 27, 'middle')
    d.matrix(50, 312, 100, 100, 1, 1, 'green', [['S₁₂₁']], 'Coupling', '64 × 64')
    d.text(195, 370, '×', 36, 'middle')
    d.matrix(235, 312, 230, 100, 1, 1, 'purple', [['Z₂₁ᵀ']], 'Inner factor', '64 × 256')
    d.text(511, 370, '=', 36, 'middle')
    d.matrix(555, 312, 230, 100, 1, 1, 'teal', [['T₁₂₁']], 'Task result', '64 × 256')
    d.arrow(800, 362, 875, 362)
    d.text(837, 299, 'write', 19, 'middle', color='#0f766e')
    d.matrix(880, 272, 270, 180, 3, 3, 'teal',
             [['T₁₁₁', 'T₁₁₂', 'T₁₁₃'], ['T₁₂₁', 'T₁₂₂', 'T₁₂₃'], ['T₁₃₁', 'T₁₃₂', 'T₁₃₃']],
             'Workspace T₁', '192 × 768', highlight=(1, 0), col_labels=True)
    d.text(1015, 510, 'rows: (a, k) · columns: (n, j)', 18, 'middle', color=MUTED)
    d.metrics([('18 independent GEMMs', 'tasks indexed by (i, k, j)'),
               ('(M, N, K) = (64, 256, 64)', 'reduce the B-rank coordinate b'),
               ('No intermediate gather', 'each task writes its final location')])
    d.save()


def stage3():
    d = Diagram('tlr_stage3_reduction', 'Stage 3 · Fold the tile sum into GEMM',
                'Fix i=1. Consume all three contraction tiles in one dense-output update.',
                'The outer factors U11, U12, U13 form a 256 by 192 panel. The terminal workspace T1 is 192 by 768. One GEMM writes the three dense output tiles C11, C12, C13 and applies beta once.')
    d.text(600, 145, 'C₁,J ← α U₁ T₁ + β C₁,J', 27, 'middle')
    d.matrix(60, 244, 270, 240, 1, 3, 'blue', [['U₁₁', 'U₁₂', 'U₁₃']], 'Outer panel U₁', '256 × 192')
    d.text(375, 373, '×', 38, 'middle')
    d.matrix(420, 244, 330, 240, 3, 3, 'teal',
             [['T₁₁₁', 'T₁₁₂', 'T₁₁₃'], ['T₁₂₁', 'T₁₂₂', 'T₁₂₃'], ['T₁₃₁', 'T₁₃₂', 'T₁₃₃']],
             'Terminal matrix T₁', '192 × 768', col_labels=True)
    d.text(810, 373, '→', 38, 'middle')
    d.matrix(870, 244, 270, 240, 1, 3, 'gold', [['C₁₁', 'C₁₂', 'C₁₃']], 'Dense output C₁,J', '256 × 768')
    d.text(585, 533, 'Match each U column segment to the corresponding k row segment of T.', 21, 'middle', color=MUTED)
    d.metrics([('Reduction K = 3 × 64 = 192', 'the paired coordinate (a, k)'),
               ('(M, N, K) = (256, 768, 192)', 'three output columns in one panel'),
               ('2 final GEMMs', 'one per output tile row i')])
    d.save()


def transpose():
    d = Diagram('tlr_stage1_transposed_blocks', 'Stage 1 caveat · Batch with a shared left operand',
                'If the right factors are stored as consecutive transposed blocks Rⱼ = Wₖⱼᵀ.',
                'Three separate 64 by 256 column-major right blocks are consumed transposed. Three coupling GEMMs reuse the same 128 by 256 left panel. This particular block layout prevents ordinary column fusion; it is not forced by a TLR transpose flag.')
    d.text(600, 145, 'Sₖ,ⱼ = Aₖ Rⱼᵀ     with     Aₖ = [V₁ₖᵀ; V₂ₖᵀ]', 27, 'middle')
    d.matrix(60, 276, 285, 144, 2, 1, 'blue', [['V₁ₖᵀ'], ['V₂ₖᵀ']], 'One shared left panel', '128 × 256')
    for idx, y in enumerate([228, 336, 444], 1):
        d.text(398, y + 42, '×', 31, 'middle')
        d.rect(440, y, 255, 70, *COLORS['orange'])
        d.text(567, y + 45, f'R{str(idx).translate(str.maketrans("123", "₁₂₃"))}ᵀ : 256 × 64', 23, 'middle')
        d.text(735, y + 43, '=', 31, 'middle')
        d.rect(778, y, 320, 70, *COLORS['green'])
        sub = str(idx).translate(str.maketrans('123', '₁₂₃'))
        d.text(938, y + 44, f'[S₁ₖ{sub}; S₂ₖ{sub}] : 128 × 64', 22, 'middle')
    d.metrics([('1 wide GEMM → 3 tasks', 'same arithmetic; different grouping'),
               ('Right layout: (n, j):(1, NK)', 'column fusion blocked by strides'),
               ('Same left operand pointer', 'reuse can help preserve throughput')])
    d.save()


if __name__ == '__main__':
    stage1()
    stage2()
    stage3()
    transpose()
