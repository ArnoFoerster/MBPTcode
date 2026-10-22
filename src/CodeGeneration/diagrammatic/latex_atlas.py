"""LaTeX (TikZ) atlas of the Goldstone diagrams + the missing-in-ISR engine.

Truth-in-rendering: every drawing comes from the same Skeleton/ordering
objects that pass the exact-oracle gates (goldstone.py); nothing is drawn
that the engine does not evaluate.

Missing-in-ISR criterion (established 2026-08-07): ISR-ADC(n) carries
configuration manifolds up to k_max = floor(n/2), i.e. omega-cuts of at
most 2*k_max + 1 lines.  A time ordering with a wider omega-cut cannot be
hosted -- its group-resummed content is exactly the delta = diagram - ISR
correction.  Order 3: max omega-cut is 3 lines -> NOTHING missing (gated
below).  Order 4: the seven-line-cut App-A quartet class.  Order 5: the
same criterion selects >= 7-line cuts (ADC(5) still stops at k2).

Usage:
  python -m src.CodeGeneration.diagrammatic.latex_atlas \
      --order 4 [--missing-only] [--out atlas_order4.tex]
One tikzpicture per time ordering (missing-only) or per skeleton with its
ordering count (full atlas).  Time runs upward; particle lines solid with
up-arrows, hole lines dashed with down-arrows; omega-cuts annotated.
"""
import argparse

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.Base.pdaggerq_interface import OCC_LETTERS
from src.CodeGeneration.diagrammatic.goldstone import (
    _BLOCKS, build_skeletons, ordering_sector, _ordering_cuts)


def missing_orderings(skel, order):
    """Time orderings of one skeleton that ISR-ADC(order) cannot host."""
    width = 2 * (order // 2) + 1
    return [o for o in skel.orderings
            if any(d != 0 and len(h) + len(p) > width
                   for d, h, p in _ordering_cuts(skel, o))]


def _tikz_ordering(skel, ordering, caption):
    """One time ordering as a tikzpicture (time upward)."""
    pos = {v: i for i, v in enumerate(ordering)}
    lanes = {ln.idx: i for i, ln in enumerate(skel.lines)}
    n_lane = max(len(skel.lines), 1)
    xs = lambda i: 0.7 * i
    ys = lambda t: 1.1 * t
    out = ["\\begin{tikzpicture}[baseline,scale=0.9]"]
    # vertex bars
    for v in range(skel.order):
        tv = pos[v]
        cols = [lanes[ln.idx] for ln in skel.lines if v in (ln.src, ln.dst)]
        x0, x1 = xs(min(cols)) - 0.25, xs(max(cols)) + 0.25
        out.append(f"\\draw[line width=1.6pt] ({x0:.2f},{ys(tv):.2f}) -- "
                   f"({x1:.2f},{ys(tv):.2f});")
        if v == skel.bra_vertex:
            out.append(f"\\draw[->] ({x1:.2f},{ys(tv):.2f}) -- "
                       f"++(0.55,0) node[right] "
                       f"{{$\\scriptstyle {skel.bra_label}$}};")
        if v == skel.ket_vertex:
            out.append(f"\\draw[<-] ({x0:.2f},{ys(tv):.2f}) -- "
                       f"++(-0.55,0) node[left] "
                       f"{{$\\scriptstyle {skel.ket_label}$}};")
    # internal lines: particle solid (arrow up), hole dashed (arrow down)
    for ln in skel.lines:
        x = xs(lanes[ln.idx])
        y_src, y_dst = ys(pos[ln.src]), ys(pos[ln.dst])
        mid = 0.5 * (y_src + y_dst)
        if ln.kind == 'p':
            out.append(f"\\draw[-{{Stealth}}] ({x:.2f},{y_src:.2f}) -- "
                       f"({x:.2f},{mid:.2f});")
            out.append(f"\\draw ({x:.2f},{mid:.2f}) -- ({x:.2f},{y_dst:.2f}) "
                       f"node[midway,right] {{$\\scriptstyle {ln.idx}$}};")
        else:
            out.append(f"\\draw[dashed,-{{Stealth}}] ({x:.2f},{y_src:.2f}) "
                       f"-- ({x:.2f},{mid:.2f});")
            out.append(f"\\draw[dashed] ({x:.2f},{mid:.2f}) -- "
                       f"({x:.2f},{y_dst:.2f}) "
                       f"node[midway,right] {{$\\scriptstyle {ln.idx}$}};")
    # omega-cut annotations
    for k, (d, h, p) in enumerate(_ordering_cuts(skel, ordering)):
        y = ys(k) + 0.55
        tag = (f"$\\omega^{{{'+' if d > 0 else '-'}}}$ {len(h)}h{len(p)}p"
               if d != 0 else f"{len(h)}h{len(p)}p")
        out.append(f"\\draw[dotted,gray] (-0.8,{y:.2f}) -- "
                   f"({xs(n_lane - 1) + 0.8:.2f},{y:.2f}) "
                   f"node[right,black] {{\\tiny {tag}}};")
    out.append("\\end{tikzpicture}")
    out.append(f"\\par\\smallskip{{\\tiny {caption}}}")
    return '\n'.join(out)


def _formula(skel, ordering):
    """Explicit algebraic value of one time ordering (the engine's value
    rule, rendered): w * sum_dummies prod<..||..> / prod(cut denominators)."""
    from fractions import Fraction
    from src.CodeGeneration.diagrammatic.goldstone import (
        ordering_sector, ordering_sign)
    sec = ordering_sector(skel, ordering)
    w = Fraction(skel.coeff / skel.n_ext
                 * ordering_sign(skel, sec)).limit_denominator(720)
    sgn = '-' if w < 0 else '+'
    aw = abs(w)
    pre = (f"{sgn}\\tfrac{{{aw.numerator}}}{{{aw.denominator}}}"
           if aw.denominator != 1 else f"{sgn}{aw.numerator}")
    dummies = sorted({ln.idx for ln in skel.lines})
    num = ''.join(
        f"\\langle {c[0]}{c[1]}\\|{a[0]}{a[1]}\\rangle"
        for c, a in skel.factors)

    def eps(lbl):
        return f"\\varepsilon_{{{lbl}}}"

    dens = []
    for d, h, p in _ordering_cuts(skel, ordering):
        terms = []
        if d != 0:
            terms.append('\\omega' if d > 0 else '-\\omega')
        terms += [f"+{eps(l)}" for l in h]
        terms += [f"-{eps(l)}" for l in p]
        s = ''.join(terms)
        dens.append(f"({s.lstrip('+')})")
    return (f"$\\displaystyle {pre}\\!\\!\\sum_{{{''.join(dummies)}}}"
            f"\\frac{{{num}}}{{{''.join(dens)}}}$")


def _resummed_formula(grp):
    """Closed form of one quartet group (all four orderings summed, 4p3h
    pole cancelled):  w * sum N * (u+E_M) / ((u+E_A)(u+E_B) E_f E_f')."""
    from fractions import Fraction
    from src.CodeGeneration.diagrammatic.goldstone import (
        ordering_sign)
    skel = grp.skel
    w = Fraction(skel.coeff / skel.n_ext
                 * ordering_sign(skel, grp.sector)).limit_denominator(720)
    sgn = '-' if w < 0 else '+'
    aw = abs(w)
    pre = (f"{sgn}\\tfrac{{{aw.numerator}}}{{{aw.denominator}}}"
           if aw.denominator != 1 else f"{sgn}{aw.numerator}")
    dummies = sorted({ln.idx for ln in skel.lines})
    num = ''.join(f"\\langle {c[0]}{c[1]}\\|{a[0]}{a[1]}\\rangle"
                  for c, a in skel.factors)

    def esum(h, p, omega=None):
        terms = ([omega] if omega else [])
        terms += [f"+\\varepsilon_{{{l}}}" for l in h]
        terms += [f"-\\varepsilon_{{{l}}}" for l in p]
        return ''.join(terms).lstrip('+')

    dA, hA, pA = grp.cuts[0]
    _, hM, pM = grp.cuts[1]
    _, hB, pB = grp.cuts[2]
    om = '\\omega' if dA > 0 else '-\\omega'
    hf = [l for l in hM if l not in hA]
    pf = [l for l in pM if l not in pA]
    hfp = [l for l in hM if l not in hB]
    pfp = [l for l in pM if l not in pB]
    A = esum(hA, pA, om)
    B = esum(hB, pB, om)
    F = esum(hf, pf)
    Fp = esum(hfp, pfp)
    s = f"{pre}\\tfrac12\\!\\sum_{{{''.join(dummies)}}}{num}"
    up = (f"$\\displaystyle {s}\\,"
          f"\\Big[\\frac{{1}}{{{A}}}+\\frac{{1}}{{{B}}}\\Big]"
          f"\\frac{{1}}{{({F})({Fp})}}$")
    cp = (f"$\\displaystyle {s}\\,"
          f"\\Big[\\frac{{1}}{{{F}}}+\\frac{{1}}{{{Fp}}}\\Big]"
          f"\\frac{{1}}{{({A})({B})}}$")
    return (f"$U$-piece:\\par\\smallskip {up}\\par\\smallskip "
            f"$C$-piece:\\par\\smallskip {cp}")


_PRE = """\\documentclass[10pt]{article}
\\usepackage[margin=1.5cm]{geometry}
\\usepackage{amsmath}
\\usepackage{tikz}
\\usetikzlibrary{arrows.meta}
\\setlength{\\parindent}{0pt}
\\begin{document}
"""


def _esc(s):
    return s.replace('<', '$\\langle$').replace('>', '$\\rangle$') \
            .replace('||', '\\,\\|\\,').replace('_', '\\_')


def emit_atlas(order, path, missing_only=False):
    out = [_PRE]
    title = (f"Goldstone diagrams of $M(\\omega;{order})$"
             + (" missing from ISR-ADC(%d)" % order if missing_only else ""))
    out.append(f"\\section*{{{title}}}")
    total = 0
    for block in _BLOCKS:
        skels, census = build_skeletons(order, block)
        items = []
        for skel in skels:
            sel = (missing_orderings(skel, order) if missing_only
                   else skel.orderings)
            for o in sel:
                sec = ordering_sector(skel, o)
                cap = (f"ordering {o}, sector {'I' if sec > 0 else 'II'}")
                items.append(_tikz_ordering(skel, o, cap)
                             + "\n\\par\\smallskip\n"
                             + _formula(skel, o))
        total += len(items)
        out.append(f"\\subsection*{{block ({block[0]},{block[1]}): "
                   f"{len(items)} orderings}}")
        for it in items:
            out.append(it + "\n\\par\\bigskip\\hrule\\bigskip")
    out.append("\\end{document}")
    with open(path, 'w') as fh:
        fh.write('\n'.join(out))
    return total


def emit_resummed(order, path):
    """One entry per quartet group: representative diagram (O1) + the
    resummed closed form (4p3h pole cancelled, sum of all 4 orderings)."""
    from src.CodeGeneration.diagrammatic.quartet_pf import (
        quartet_groups)
    out = [_PRE, f"\\section*{{Resummed quartet groups of $M(\\omega;{order})$"
           " --- the class beyond the ISR representation (sum of the four"
           " time orderings each; the 4p3h/4h3p pole is cancelled)}"]
    total = 0
    for block in _BLOCKS:
        groups = quartet_groups(block, order)
        out.append(f"\\subsection*{{block ({block[0]},{block[1]}): "
                   f"{len(groups)} groups}}")
        for grp in groups:
            cap = ("representative ordering O1 of 4; "
                   f"sector {'I' if grp.sector > 0 else 'II'}")
            out.append(_tikz_ordering(grp.skel, grp.orderings['O1'], cap)
                       + "\n\\par\\smallskip\n" + _resummed_formula(grp)
                       + "\n\\par\\bigskip\\hrule\\bigskip")
            total += 1
    out.append("\\end{document}")
    with open(path, 'w') as fh:
        fh.write('\n'.join(out))
    return total


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--order', type=int, default=4)
    ap.add_argument('--missing-only', action='store_true')
    ap.add_argument('--resummed', action='store_true',
                    help='one closed-form entry per quartet group')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    path = a.out or (f"goldstone_order{a.order}"
                     + ("_resummed" if a.resummed else
                        "_missing" if a.missing_only else "") + ".tex")
    if a.resummed:
        n = emit_resummed(a.order, path)
        print(f"{n} groups -> {path}")
    else:
        n = emit_atlas(a.order, path, missing_only=a.missing_only)
        print(f"{n} orderings -> {path}")
