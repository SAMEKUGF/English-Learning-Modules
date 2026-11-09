#!/usr/bin/env python3
"""Rend une vidéo « Apprendre l'anglais en chansons » à partir de lesson.json.

Usage :
    python3 render.py --audio chemin/vers/chanson.mp3 --out lecon.mp4 [--preview 41.5]

Structure de la vidéo :
  1. Intro animée
  2. Écoute & traduction : karaoké mot à mot + traduction française phrase par phrase
  3. Vocabulaire : cartes animées, prononciation ×3 avec pause « Répétez ! »
  4. Révision : même extrait, mots clés masqués (texte à trous) révélés au chant
  5. Outro récapitulative
"""
import argparse, json, math, os, subprocess, sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, SR = 1920, 1080, 30, 44100
HERE = os.path.dirname(os.path.abspath(__file__))
FI = "/usr/share/fonts/opentype/inter/"
FD = "/usr/share/fonts/truetype/dejavu/"

# palette
GOLD = (255, 196, 87)
TEAL = (94, 234, 212)
ROSE = (255, 122, 162)
LAV = (184, 196, 255)
WHITE = (255, 255, 255)

CARD_LEN = 9.0          # durée d'une carte de vocabulaire
SECTION_TITLE = 3.0     # titre de la partie vocabulaire
TTS_AT = (1.2, 3.9, 6.6)  # moments où la prononciation est jouée dans une carte


# ---------------------------------------------------------------- utilitaires
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_back(x):
    x = clamp(x)
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def window(t, a, b, fin=0.4, fout=0.4):
    """Opacité 0→1→0 sur [a, b]."""
    if t < a or t > b:
        return 0.0
    return min(ease_out((t - a) / fin) if fin else 1, ease_out((b - t) / fout) if fout else 1)


_fonts = {}


def font(name, size):
    key = (name, size)
    if key not in _fonts:
        path = name if name.startswith("/") else (FD + name if name.startswith("DejaVu") else FI + name)
        _fonts[key] = ImageFont.truetype(path, size)
    return _fonts[key]


def tokens(s):
    return [w.strip(".,!?;:'\"").lower() for w in s.split()]


# ---------------------------------------------------------------- timeline
PRE, POST = 0.5, 0.6
TEACH_PRE = 0.5          # silence avant la voix du professeur
G_TEACH_DUR = {}          # durée de chaque clip professeur (remplie au démarrage)   # marge audio avant / après chaque phrase


def build_timeline(L):
    """Segments : (type, début, fin, info).
    Pour chaque phrase : chant original → cartes des mots avancés (×3) → réécoute."""
    segs = []
    t = 0.0

    def add(kind, dur, **info):
        nonlocal t
        segs.append((kind, t, t + dur, info))
        t += dur

    a, b = L["intro_audio"]
    add("intro", b - a)
    used = set()
    n = len(L["lines"])
    for i, ln in enumerate(L["lines"]):
        a0, a1 = ln["start"] - PRE, ln["end"] + POST
        add("phrase", a1 - a0, line=i, a0=a0, rep=0)
        if ln.get("teacher"):
            add("teacher", G_TEACH_DUR[i] + TEACH_PRE + 0.9, line=i)
        cards = []
        for _, vid in ln.get("keys", []):
            if vid not in used:
                cards.append(vid)
                used.add(vid)
        if i == n - 1:  # les mots restants (ex. le titre) après la dernière phrase
            cards += [k for k in range(len(L["vocab"])) if k not in used]
            used.update(cards)
        for vid in cards:
            add("card", CARD_LEN, vid=vid, line=i)
        if cards:
            add("phrase", a1 - a0, line=i, a0=a0, rep=1)
    a, b = L["review_audio"]
    add("review", b - a, a0=a)
    return segs, t


def prepare_lines(L):
    for ln in L["lines"]:
        words = ln["en"].split()
        weights = [len(w.strip(",.")) + 2.5 for w in words]
        tot = sum(weights)
        span = (ln["end"] - ln["start"]) * 0.92
        acc = ln["start"]
        times = []
        for wgt in weights:
            d = span * wgt / tot
            times.append((acc, acc + d))
            acc += d
        ln["wt"] = times
        # repère les mots clés (indices de mots)
        toks = tokens(ln["en"])
        keymap = {}
        for phrase, vid in ln.get("keys", []):
            p = tokens(phrase)
            for i in range(len(toks) - len(p) + 1):
                if toks[i:i + len(p)] == p:
                    for j in range(i, i + len(p)):
                        keymap[j] = (vid, i, i + len(p))
                    break
        ln["keymap"] = keymap


# ---------------------------------------------------------------- audio
def decode(path, ss=None, to=None):
    cmd = ["ffmpeg", "-v", "error"]
    if ss is not None:
        cmd += ["-ss", str(ss)]
    if to is not None:
        cmd += ["-t", str(to - (ss or 0))]
    cmd += ["-i", path, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def fade(x, fin=0.0, fout=0.0):
    n = len(x)
    if fin:
        k = min(n, int(fin * SR))
        x[:k] *= np.linspace(0, 1, k)[:, None]
    if fout:
        k = min(n, int(fout * SR))
        x[n - k:] *= np.linspace(1, 0, k)[:, None]
    return x


def place(buf, clip, t):
    i = int(t * SR)
    j = min(len(buf), i + len(clip))
    if j > i:
        buf[i:j] += clip[: j - i]


def build_audio(L, segs, total, song_path):
    buf = np.zeros((int(total * SR) + SR, 2), dtype=np.float32)
    song = decode(song_path)

    def cut(a, b):
        return song[int(a * SR):int(b * SR)].copy()

    a, b = L["bed_audio"]
    bed = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(b - a), "-i", song_path,
         "-af", "lowpass=f=900,volume=0.16", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
        capture_output=True, check=True).stdout
    bed = np.frombuffer(bed, dtype=np.float32).reshape(-1, 2)
    tts = [decode(os.path.join(HERE, v["tts"])) * 1.1 for v in L["vocab"]]

    bed_start = None
    for k, (kind, s0, s1, info) in enumerate(segs):
        if kind == "intro":
            a, b = L["intro_audio"]
            place(buf, fade(cut(a, b), 1.0, 1.2), s0)
        elif kind == "phrase":
            a0 = info["a0"]
            place(buf, fade(cut(a0, a0 + s1 - s0), 0.35, 0.45), s0)
        elif kind == "review":
            a0 = info["a0"]
            place(buf, fade(cut(a0, a0 + s1 - s0), 0.6, 2.5), s0)
        elif kind == "teacher":
            ln = L["lines"][info["line"]]
            place(buf, decode(os.path.join(HERE, ln["teacher"])) * 1.15, s0 + TEACH_PRE)
            need = int((s1 - s0) * SR)
            reps = int(np.ceil(need / len(bed)))
            place(buf, fade(np.concatenate([bed] * reps)[:need].copy(), 0.4, 0.4) * 0.8, s0)
        elif kind == "card":
            for at in TTS_AT:
                place(buf, tts[info["vid"]], s0 + at)
            if bed_start is None:
                bed_start = s0
            nxt = segs[k + 1][0] if k + 1 < len(segs) else None
            if nxt != "card":  # fin d'un groupe de cartes : fond musical doux
                need = int((s1 - bed_start) * SR)
                reps = int(np.ceil(need / len(bed)))
                place(buf, fade(np.concatenate([bed] * reps)[:need].copy(), 0.6, 0.6), bed_start)
                bed_start = None
    np.clip(buf, -1, 1, out=buf)
    return buf[: int(total * SR)]


def spectrum(buf, total, nb=56):
    mono = buf.mean(axis=1)
    n = 2048
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    edges = np.geomspace(55, 9000, nb + 1)
    idx = [(np.searchsorted(freqs, edges[i]), max(np.searchsorted(freqs, edges[i + 1]), np.searchsorted(freqs, edges[i]) + 1)) for i in range(nb)]
    nf = int(total * FPS)
    out = np.zeros((nf, nb), dtype=np.float32)
    prev = np.zeros(nb)
    for f in range(nf):
        c = int(f / FPS * SR)
        seg = mono[max(0, c - n // 2): c + n // 2]
        if len(seg) < n:
            seg = np.pad(seg, (0, n - len(seg)))
        mag = np.abs(np.fft.rfft(seg * win))
        v = np.array([mag[a:b].mean() for a, b in idx])
        v = np.log1p(v * 4) / 4.2
        v = np.clip(v * (0.75 + np.linspace(0, 0.7, nb)), 0, 1)
        prev = np.where(v > prev, v * 0.7 + prev * 0.3, prev * 0.86 + v * 0.14)
        out[f] = prev
    level = out.mean(axis=1)
    return out, level


# ---------------------------------------------------------------- dessin
G = {}  # état global partagé avec les workers


def background(t, energy):
    gw, gh = 192, 108
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    base = np.zeros((gh, gw, 3), np.float32)
    top = np.array([12, 14, 38], np.float32)
    bot = np.array([26, 12, 44], np.float32)
    g = (yy / gh)[..., None]
    base += top * (1 - g) + bot * g
    blobs = [
        ((0.25, 0.30), (0.08, 0.05), 0.33, (110, 70, 220)),
        ((0.78, 0.65), (0.06, 0.07), 0.30, (20, 150, 160)),
        ((0.55, 0.15), (0.05, 0.06), 0.26, (220, 90, 140)),
        ((0.15, 0.85), (0.07, 0.04), 0.24, (230, 160, 60)),
    ]
    for k, ((cx, cy), (ax, ay), r, col) in enumerate(blobs):
        px = (cx + ax * math.sin(t * 0.21 + k * 1.7)) * gw
        py = (cy + ay * math.cos(t * 0.17 + k * 2.3)) * gh
        rr = r * gw * (1 + 0.12 * energy)
        d2 = ((xx - px) ** 2 + (yy - py) ** 2) / (rr * rr)
        base += np.exp(-d2 * 2.2)[..., None] * np.array(col, np.float32) * (0.36 + 0.25 * energy)
    # vignette
    v = 1 - 0.45 * (((xx / gw - 0.5) ** 2 + (yy / gh - 0.5) ** 2) * 2.2)
    base *= v[..., None]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    return img.convert("RGBA")


def rgba(c, a):
    return (c[0], c[1], c[2], int(255 * clamp(a)))


def text_center(d, xy, s, f, fill, anchor="mm"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def pill(d, box, fill, outline=None, r=None, width=2):
    r = r if r is not None else (box[3] - box[1]) // 2
    d.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def draw_chrome(img, d, t, f, sect_name, sect_col, L):
    # logo
    pill(d, (60, 44, 470, 104), (255, 255, 255, 18), (255, 255, 255, 40))
    # note de musique dessinée
    d.ellipse((84, 78, 102, 94), fill=GOLD + (255,))
    d.rectangle((98, 54, 102, 86), fill=GOLD + (255,))
    d.polygon([(102, 54), (116, 60), (116, 66), (102, 61)], fill=GOLD + (255,))
    d.text((130, 74), "English through Songs", font=font("Inter-SemiBold.otf", 28), fill=(255, 255, 255, 235), anchor="lm")
    # titre de la chanson
    d.text((W - 60, 62), L["title"], font=font("Inter-SemiBold.otf", 26), fill=(255, 255, 255, 220), anchor="rm")
    d.text((W - 60, 94), L["artist"] + "   ·   Niveau " + L["level"], font=font("Inter-Regular.otf", 22), fill=(255, 255, 255, 140), anchor="rm")
    # badge de section
    if sect_name:
        fw = font("Inter-Bold.otf", 24)
        tw = d.textlength(sect_name, font=fw)
        x0 = 60
        pill(d, (x0, 124, x0 + tw + 56, 168), sect_col + (40,), sect_col + (160,))
        d.ellipse((x0 + 18, 140, x0 + 30, 152), fill=sect_col + (255,))
        d.text((x0 + 40, 146), sect_name, font=fw, fill=(255, 255, 255, 240), anchor="lm")


def draw_spectrum(d, fidx, alpha=1.0):
    spec = G["spec"][min(fidx, len(G["spec"]) - 1)]
    nb = len(spec)
    bw, gap = 14, 8
    total = nb * (bw + gap) - gap
    x0 = (W - total) / 2
    base = H - 92
    for i, v in enumerate(spec):
        h = 6 + v * 70
        x = x0 + i * (bw + gap)
        u = i / (nb - 1)
        col = tuple(int(TEAL[k] * (1 - u) + ROSE[k] * u) for k in range(3))
        d.rounded_rectangle((x, base - h, x + bw, base), 5, fill=col + (int(150 * alpha),))
        d.rounded_rectangle((x, base + 6, x + bw, base + 6 + h * 0.25), 4, fill=col + (int(40 * alpha),))


def draw_progress(d, t):
    total = G["total"]
    x0, x1, y = 60, W - 60, H - 34
    d.rounded_rectangle((x0, y - 3, x1, y + 3), 3, fill=(255, 255, 255, 40))
    xp = x0 + (x1 - x0) * t / total
    d.rounded_rectangle((x0, y - 3, xp, y + 3), 3, fill=GOLD + (230,))
    for name, a, b, info in G["segs"][1:]:
        if not ((name == "phrase" and info["rep"] == 0) or name == "review"):
            continue
        xm = x0 + (x1 - x0) * a / total
        d.ellipse((xm - 6, y - 6, xm + 6, y + 6), fill=(255, 255, 255, 200) if t >= a else (255, 255, 255, 70))
    d.ellipse((xp - 9, y - 9, xp + 9, y + 9), fill=(255, 255, 255, 255))


def layout_words(words, f, maxw, d):
    space = d.textlength(" ", font=f)
    rows, cur, cw = [], [], 0
    for i, w in enumerate(words):
        ww = d.textlength(w, font=f)
        if cur and cw + space + ww > maxw:
            rows.append((cur, cw))
            cur, cw = [], 0
        cur.append((i, w, ww))
        cw += (space if len(cur) > 1 else 0) + ww
    if cur:
        rows.append((cur, cw))
    return rows, space


def draw_lyric(img, t_song, ln, idx, n, appear, alpha, mode, d_fr_alpha=1.0, tag=""):
    """mode = 'listen' (karaoké + FR)  |  'review' (texte à trous)"""
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    f = font("InterDisplay-Bold.otf", 66)
    words = ln["en"].split()
    rows, space = layout_words(words, f, 1500, d)
    lh = 88
    yoff = (1 - appear) * 50
    y0 = 430 - (len(rows) - 1) * lh / 2 + yoff
    # numéro de phrase
    d.text((W / 2, y0 - 92), f"PHRASE {idx + 1} / {n}{tag}", font=font("Inter-SemiBold.otf", 22), fill=(GOLD if tag else (255, 255, 255)) + (int((230 if tag else 130) * alpha),), anchor="mm")
    for r, (row, rw) in enumerate(rows):
        x = W / 2 - rw / 2
        y = y0 + r * lh
        for (i, w, ww) in row:
            a, b = ln["wt"][i]
            prog = clamp((t_song - a) / max(0.05, b - a))
            key = ln["keymap"].get(i)
            if key:
                kcol = TEAL
            else:
                kcol = WHITE
            if mode == "review" and key:
                # mot masqué jusqu'à ce qu'il soit chanté
                vid, k0, k1 = key
                ka = ln["wt"][k0][0]
                rev = ease_back((t_song - ka) / 0.45)
                bw_ = ww
                if rev <= 0:
                    d.rounded_rectangle((x - 4, y - 30, x + bw_ + 4, y + 34), 10, fill=(255, 255, 255, int(28 * alpha)), outline=GOLD + (int(200 * alpha),), width=3)
                    d.text((x + bw_ / 2, y + 2), "?", font=font("InterDisplay-Bold.otf", 44), fill=GOLD + (int(220 * alpha),), anchor="mm")
                else:
                    sc = 0.6 + 0.4 * rev
                    fs = font("InterDisplay-Bold.otf", max(10, int(66 * sc)))
                    d.rounded_rectangle((x - 6, y - 34, x + ww + 6, y + 38), 12, fill=GOLD + (int(40 * alpha * clamp(rev)),))
                    d.text((x + ww / 2, y + 2), w, font=fs, fill=GOLD + (int(255 * alpha),), anchor="mm")
                x += ww + space
                continue
            if mode == "listen":
                base_a = 0.38 + 0.62 * prog
                col = GOLD if (key and prog > 0) else kcol
                if 0 < prog < 1:
                    col = GOLD
                lift = -6 * math.sin(math.pi * prog) if 0 < prog < 1 else 0
            else:
                base_a = 0.45 + 0.55 * prog
                col = WHITE
                lift = -6 * math.sin(math.pi * prog) if 0 < prog < 1 else 0
            d.text((x, y + lift), w, font=f, fill=col + (int(255 * alpha * base_a),), anchor="lm")
            if key and mode == "listen":
                d.rounded_rectangle((x, y + 40, x + ww, y + 46), 3, fill=TEAL + (int(220 * alpha),))
            x += ww + space
    # traduction française
    if d_fr_alpha > 0:
        ff = font("Inter-MediumItalic.otf", 44)
        frows, fsp = layout_words(ln["fr"].split(), ff, 1500, d)
        fy = y0 + len(rows) * lh + 40
        a_fr = alpha * d_fr_alpha
        # trait décoratif
        d.rounded_rectangle((W / 2 - 40, fy - 12, W / 2 + 40, fy - 8), 2, fill=ROSE + (int(200 * a_fr),))
        d.text((W / 2, fy + 18), "FR", font=font("Inter-Bold.otf", 18), fill=ROSE + (int(220 * a_fr),), anchor="mm")
        for r, (row, rw) in enumerate(frows):
            txt = " ".join(w for _, w, _ in row)
            d.text((W / 2, fy + 70 + r * 60 + (1 - d_fr_alpha) * 20), txt, font=ff, fill=LAV + (int(255 * a_fr),), anchor="mm")
    img.alpha_composite(lay)


def section_title(img, local, dur, kicker, title, sub, col):
    a = window(local, 0, dur, 0.5, 0.5)
    if a <= 0:
        return
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    d.rectangle((0, 0, W, H), fill=(8, 8, 24, int(150 * a)))
    p = ease_out(local / 0.8)
    lw = 360 * p
    d.rounded_rectangle((W / 2 - lw / 2, 430, W / 2 + lw / 2, 436), 3, fill=col + (int(255 * a),))
    d.text((W / 2, 395), kicker, font=font("Inter-Bold.otf", 28), fill=col + (int(255 * a),), anchor="mm")
    sh = (1 - p) * 40
    d.text((W / 2, 520 + sh), title, font=font("InterDisplay-Black.otf", 96), fill=(255, 255, 255, int(255 * a)), anchor="mm")
    d.text((W / 2, 615 + sh * 1.5), sub, font=font("Inter-Regular.otf", 36), fill=(255, 255, 255, int(190 * a)), anchor="mm")
    img.alpha_composite(lay)


def draw_intro(img, t, L):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    out = 1 - ease_out((t - 6.2) / 0.8)
    # sous-titre
    a1 = ease_out((t - 0.3) / 0.8) * out
    d.text((W / 2, 290), "APPRENDRE L'ANGLAIS EN CHANSONS", font=font("Inter-Bold.otf", 30), fill=GOLD + (int(255 * a1),), anchor="mm")
    # titre lettre par lettre
    title = L["title"]
    ft = font("InterDisplay-Black.otf", 104)
    tw = d.textlength(title, font=ft)
    x = W / 2 - tw / 2
    for i, ch in enumerate(title):
        p = ease_back((t - 0.6 - i * 0.035) / 0.5)
        cw = d.textlength(ch, font=ft)
        if p > 0:
            d.text((x, 420 + (1 - p) * 60), ch, font=ft, fill=(255, 255, 255, int(255 * clamp(p) * out)), anchor="lm")
        x += cw
    a2 = ease_out((t - 1.8) / 0.8) * out
    d.text((W / 2, 520), "— " + L["artist"] + " —", font=font("Inter-MediumItalic.otf", 42), fill=LAV + (int(255 * a2),), anchor="mm")
    # plan de la leçon
    steps = [("1", "Phrase chantée + traduction", TEAL), ("2", "Mots avancés : répétez ×3", GOLD), ("3", "Réécoute + texte à trous", ROSE)]
    for k, (num, lab, col) in enumerate(steps):
        p = ease_out((t - 2.6 - k * 0.35) / 0.6) * out
        if p <= 0:
            continue
        cx = W / 2 + (k - 1) * 570
        y = 690 + (1 - p) * 30
        pill(d, (cx - 265, y - 44, cx + 265, y + 44), col + (int(36 * p),), col + (int(170 * p),))
        d.ellipse((cx - 245, y - 26, cx - 193, y + 26), fill=col + (int(255 * p),))
        d.text((cx - 219, y), num, font=font("Inter-Black.otf", 30), fill=(14, 14, 36, int(255 * p)), anchor="mm")
        d.text((cx - 175, y), lab, font=font("Inter-SemiBold.otf", 30), fill=(255, 255, 255, int(255 * p)), anchor="lm")
    img.alpha_composite(lay)


def speaker_icon(d, cx, cy, s, col, a, waves):
    d.polygon([(cx - s, cy - s * 0.4), (cx - s * 0.4, cy - s * 0.4), (cx + s * 0.2, cy - s), (cx + s * 0.2, cy + s), (cx - s * 0.4, cy + s * 0.4), (cx - s, cy + s * 0.4)], fill=col + (int(255 * a),))
    for k in range(waves):
        r = s * (0.7 + k * 0.45)
        d.arc((cx + s * 0.1 - r, cy - r, cx + s * 0.1 + r, cy + r), -45, 45, fill=col + (int(255 * a * (1 - k * 0.25)),), width=5)


def draw_vocab_card(img, local, i, v, n):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    pin = ease_out(local / 0.7)
    pout = ease_in_out((local - (CARD_LEN - 0.55)) / 0.55)
    a = pin * (1 - pout)
    dx = (1 - pin) * 260 - pout * 260
    x0, y0, x1, y1 = 200 + dx, 205, W - 200 + dx, 905
    # ombre + panneau verre
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((x0 + 10, y0 + 24, x1 + 10, y1 + 24), 40, fill=(0, 0, 0, int(110 * a)))
    sh = sh.filter(ImageFilter.GaussianBlur(24))
    img.alpha_composite(sh)
    d.rounded_rectangle((x0, y0, x1, y1), 40, fill=(255, 255, 255, int(22 * a)), outline=(255, 255, 255, int(55 * a)), width=2)
    d.rounded_rectangle((x0, y0, x0 + 12, y1), 6, fill=GOLD + (int(230 * a),))
    # en-tête
    d.text((x0 + 70, y0 + 60), f"MOT {i + 1} / {n}", font=font("Inter-Bold.otf", 24), fill=GOLD + (int(230 * a),), anchor="lm")
    fpos = font("Inter-SemiBold.otf", 24)
    tw = d.textlength(v["pos"], font=fpos)
    pill(d, (x1 - 70 - tw - 40, y0 + 38, x1 - 70, y0 + 82), TEAL + (int(36 * a),), TEAL + (int(170 * a),))
    d.text((x1 - 70 - tw / 2 - 20, y0 + 60), v["pos"], font=fpos, fill=TEAL + (int(255 * a),), anchor="mm")
    # mot (pulse à chaque prononciation)
    pulse = 0
    for k in TTS_AT:
        pulse = max(pulse, math.exp(-((local - k - 0.25) ** 2) / 0.05))
    size = 104 if len(v["word"]) < 16 else 84
    fw = font("InterDisplay-Black.otf", int(size * (1 + 0.05 * pulse)))
    wy = y0 + 175
    pw = ease_back((local - 0.25) / 0.6)
    d.text((x0 + 70, wy + (1 - clamp(pw)) * 30), v["word"], font=fw, fill=GOLD + (int(255 * a * clamp(pw)),), anchor="lm")
    # API
    pa = ease_out((local - 0.6) / 0.5) * a
    d.text((x0 + 74, wy + 88), v["ipa"], font=font("DejaVuSans.ttf", 44), fill=(255, 255, 255, int(200 * pa)), anchor="lm")
    # traduction
    pf = ease_out((local - 0.9) / 0.5) * a
    d.text((x0 + 70, wy + 185), "→", font=font("Inter-Bold.otf", 50), fill=ROSE + (int(255 * pf),), anchor="lm")
    d.text((x0 + 140, wy + 185), v["fr"], font=font("InterDisplay-SemiBold.otf", 56), fill=(255, 255, 255, int(255 * pf),), anchor="lm")
    # exemple tiré de la chanson
    pe = ease_out((local - 1.3) / 0.5) * a
    ey = wy + 290
    d.rounded_rectangle((x0 + 60, ey - 44, x1 - 60, ey + 44), 22, fill=(0, 0, 0, int(60 * pe)))
    d.text((x0 + 90, ey), "♪", font=font("DejaVuSans.ttf", 40), fill=TEAL + (int(255 * pe),), anchor="lm")
    fe = font("Inter-MediumItalic.otf", 38)
    ex = v["example"]
    low = ex.lower()
    key = v["word"].lower()
    k = low.find(key)
    xx = x0 + 140
    parts = [(ex, WHITE)] if k < 0 else [(ex[:k], WHITE), (ex[k:k + len(key)], GOLD), (ex[k + len(key):], WHITE)]
    for s, c in parts:
        if not s:
            continue
        d.text((xx, ey), s, font=fe, fill=c + (int(235 * pe),), anchor="lm")
        xx += d.textlength(s, font=fe)
    # astuce
    pt = ease_out((local - 1.6) / 0.5) * a
    d.text((x0 + 70, ey + 90), "Astuce :  " + v["tip"], font=font("Inter-Regular.otf", 32), fill=LAV + (int(230 * pt),), anchor="lm")
    # module de répétition
    ry = y1 - 70
    cur = -1
    for kk, at in enumerate(TTS_AT):
        if local >= at:
            cur = kk
    playing = any(at <= local < at + 1.3 for at in TTS_AT)
    rep_a = ease_out((local - 0.9) / 0.4) * a
    for kk in range(3):
        cx = x0 + 100 + kk * 64
        done = kk <= cur
        fill = GOLD + (int(255 * rep_a),) if done else (255, 255, 255, int(40 * rep_a))
        d.ellipse((cx - 20, ry - 20, cx + 20, ry + 20), fill=fill)
        d.text((cx, ry), str(kk + 1), font=font("Inter-Bold.otf", 22), fill=((14, 14, 36) if done else (255, 255, 255)) + (int(255 * rep_a),), anchor="mm")
    lx = x0 + 330
    if cur < 0:
        label, col = "Écoutez bien…", (255, 255, 255)
    elif playing:
        label, col = "Écoutez", TEAL
    else:
        label, col = "Répétez à voix haute !", ROSE
    if playing:
        speaker_icon(d, lx + 20, ry, 20, TEAL, rep_a, 1 + int((local * 6) % 3))
    else:
        # micro dessiné
        d.rounded_rectangle((lx + 10, ry - 22, lx + 30, ry + 6), 10, fill=col + (int(255 * rep_a),))
        d.arc((lx + 2, ry - 10, lx + 38, ry + 16), 0, 180, fill=col + (int(255 * rep_a),), width=4)
        d.rectangle((lx + 18, ry + 16, lx + 22, ry + 24), fill=col + (int(255 * rep_a),))
    d.text((lx + 64, ry), label, font=font("Inter-Bold.otf", 34), fill=col + (int(255 * rep_a),), anchor="lm")
    # anneaux de pulsation pendant la répétition
    if cur >= 0 and not playing:
        ph = (local * 1.4) % 1
        rr = 30 + ph * 40
        d.ellipse((lx + 20 - rr, ry - rr, lx + 20 + rr, ry + rr), outline=ROSE + (int(160 * (1 - ph) * rep_a),), width=3)
    d.text((x1 - 70, ry), "×3", font=font("InterDisplay-Black.otf", 54), fill=(255, 255, 255, int(70 * rep_a)), anchor="rm")
    img.alpha_composite(lay)


def draw_outro(img, local, dur, L):
    a = ease_out(local / 0.8)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    d.rectangle((0, 0, W, H), fill=(8, 8, 24, int(200 * a)))
    d.text((W / 2, 300), "Bravo !", font=font("InterDisplay-Black.otf", 110), fill=GOLD + (int(255 * a),), anchor="mm")
    d.text((W / 2, 395), "Vous avez appris " + str(len(L["vocab"])) + " expressions. Chantez avec Céline !", font=font("Inter-Medium.otf", 38), fill=(255, 255, 255, int(230 * a)), anchor="mm")
    words = [v["word"] for v in L["vocab"]]
    fw = font("Inter-SemiBold.otf", 32)
    rows = [words[:4], words[4:]]
    for r, row in enumerate(rows):
        widths = [d.textlength(w, font=fw) + 56 for w in row]
        tot = sum(widths) + 20 * (len(row) - 1)
        x = W / 2 - tot / 2
        for k, w in enumerate(row):
            p = ease_back((local - 0.4 - (r * 4 + k) * 0.12) / 0.5)
            y = 520 + r * 96 + (1 - clamp(p)) * 30
            col = [TEAL, GOLD, ROSE, LAV][(r * 4 + k) % 4]
            pill(d, (x, y - 34, x + widths[k], y + 34), col + (int(40 * clamp(p) * a),), col + (int(200 * clamp(p) * a),))
            d.text((x + widths[k] / 2, y), w, font=fw, fill=(255, 255, 255, int(255 * clamp(p) * a)), anchor="mm")
            x += widths[k] + 20
    img.alpha_composite(lay)


def load_teacher_durations(L):
    for i, ln in enumerate(L["lines"]):
        if ln.get("teacher"):
            out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                  os.path.join(HERE, ln["teacher"])], capture_output=True, text=True, check=True).stdout
            G_TEACH_DUR[i] = float(out.strip())


def draw_teacher(img, local, dur, ln, idx, n):
    """Le professeur lit la traduction : phrase anglaise rappelée en haut, français mis en avant."""
    a = window(local, 0, dur, 0.45, 0.4)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    speaking = TEACH_PRE <= local < TEACH_PRE + G_TEACH_DUR.get(idx, 0)
    # badge professeur
    cx, cy = W / 2, 250
    r = 54
    if speaking:
        for k in range(3):
            ph = ((local * 0.9) + k / 3) % 1
            rr = r + ph * 70
            d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=ROSE + (int(150 * (1 - ph) * a),), width=3)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=ROSE + (int(235 * a),))
    # tête + épaules stylisées
    d.ellipse((cx - 17, cy - 32, cx + 17, cy + 2), fill=(20, 16, 40, int(255 * a)))
    d.chord((cx - 32, cy + 6, cx + 32, cy + 62), 180, 360, fill=(20, 16, 40, int(255 * a)))
    d.text((cx, cy + r + 34), "LE PROFESSEUR TRADUIT", font=font("Inter-Bold.otf", 24), fill=ROSE + (int(240 * a),), anchor="mm")
    # rappel de la phrase anglaise
    fe = font("InterDisplay-SemiBold.otf", 40)
    rows, _ = layout_words(ln["en"].split(), fe, 1500, d)
    y = 400
    for row, rw in rows:
        d.text((W / 2, y), " ".join(w for _, w, _ in row), font=fe, fill=(255, 255, 255, int(150 * a)), anchor="mm")
        y += 54
    # traduction française, mot à mot pendant que le professeur parle
    ff = font("InterDisplay-Bold.otf", 60)
    words = ln["fr"].split()
    frows, sp = layout_words(words, ff, 1560, d)
    td = G_TEACH_DUR.get(idx, dur)
    # « Phrase X. Ça veut dire : » occupe environ 1,7 s avant la traduction
    t_fr0, t_fr1 = TEACH_PRE + 1.7, TEACH_PRE + td - 0.2
    y = y + 70
    d.rounded_rectangle((W / 2 - 40, y - 30, W / 2 + 40, y - 26), 2, fill=ROSE + (int(200 * a),))
    y += 30
    for row, rw in frows:
        x = W / 2 - rw / 2
        for (i, w, ww) in row:
            wt = t_fr0 + (t_fr1 - t_fr0) * i / max(1, len(words))
            p = clamp((local - wt) / 0.25)
            col = LAV if p >= 1 else (255, 255, 255)
            al = 0.35 + 0.65 * p
            d.text((x, y - 6 * math.sin(math.pi * p) if 0 < p < 1 else y), w, font=ff, fill=col + (int(255 * a * al),), anchor="lm")
            x += ww + sp
        y += 80
    img.alpha_composite(lay)


def render_frame(fidx):
    L, segs = G["L"], G["segs"]
    t = fidx / FPS
    energy = float(G["level"][min(fidx, len(G["level"]) - 1)])
    img = background(t, energy)
    seg = next((s for s in segs if s[1] <= t < s[2]), segs[-1])
    name, s0, s1, info = seg
    lines = L["lines"]
    n = len(lines)
    if name == "intro":
        sect = (None, TEAL)
    elif name == "phrase":
        sect = (f"Phrase {info['line'] + 1}/{n} · " + ("Écoute & traduction" if info["rep"] == 0 else "Réécoute"), TEAL)
    elif name == "teacher":
        sect = (f"Phrase {info['line'] + 1}/{n} · Le professeur traduit", ROSE)
    elif name == "card":
        sect = (f"Phrase {info['line'] + 1}/{n} · Vocabulaire ×3", GOLD)
    else:
        sect = ("Révision : texte à trous", ROSE)
    chrome = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cd = ImageDraw.Draw(chrome)
    draw_chrome(chrome, cd, t, fidx, sect[0], sect[1], L)
    draw_spectrum(cd, fidx, 0.55 if name in ("card", "teacher") else 1.0)
    draw_progress(cd, t)
    img.alpha_composite(chrome)

    loc = t - s0
    if name == "intro":
        draw_intro(img, loc, L)
    elif name == "phrase":
        ln = lines[info["line"]]
        ts = loc + info["a0"]  # temps dans la chanson
        appear = ease_out(loc / 0.45)
        alpha = appear * (1 - ease_in_out((loc - (s1 - s0 - 0.35)) / 0.35))
        if info["rep"] == 0:
            fr_a = ease_out((ts - ln["start"] - 0.3) / 0.6)
            tag = ""
        else:
            fr_a = 1.0
            tag = "  ·  RÉÉCOUTEZ ET CHANTEZ !"
        draw_lyric(img, ts, ln, info["line"], n, appear, alpha, "listen", fr_a, tag)
    elif name == "teacher":
        draw_teacher(img, loc, s1 - s0, lines[info["line"]], info["line"], n)
    elif name == "card":
        vid = info["vid"]
        draw_vocab_card(img, loc, vid, L["vocab"][vid], len(L["vocab"]))
    else:  # review : texte à trous sur tout l'extrait
        ts = loc + info["a0"]
        cur = None
        for i, ln in enumerate(lines):
            if ts >= ln["start"] - 0.5:
                cur = i
        if cur is not None:
            ln = lines[cur]
            nxt = lines[cur + 1]["start"] - 0.5 if cur + 1 < n else ln["end"] + 3.0
            appear = ease_out((ts - (ln["start"] - 0.5)) / 0.5)
            alpha = appear * (1 - ease_in_out((ts - (nxt - 0.35)) / 0.35))
            fr_a = 0.75 * ease_out((ts - (ln["end"] - 0.6)) / 0.6)
            if alpha > 0:
                draw_lyric(img, ts, ln, cur, n, appear, alpha, "review", fr_a)
        section_title(img, loc, 2.7, "RÉVISION", "Texte à trous", "Devinez les mots cachés avant qu'ils soient chantés !", ROSE)
        od = 5.5
        if t > s1 - od:
            draw_outro(img, t - (s1 - od), od, L)
    # fondu au noir initial / final
    fade_a = max(1 - ease_out(t / 0.8), ease_in_out((t - (G["total"] - 0.8)) / 0.8))
    if fade_a > 0:
        img.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, int(255 * fade_a))))
    return img.convert("RGB").tobytes()


def init_worker(state):
    G.update(state)
    G_TEACH_DUR.update(state["teach"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--lesson", default=os.path.join(HERE, "lesson.json"))
    ap.add_argument("--out", default="lecon.mp4")
    ap.add_argument("--preview", type=float, nargs="*", help="temps (s) à exporter en PNG au lieu de la vidéo")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    L = json.load(open(args.lesson, encoding="utf-8"))
    prepare_lines(L)
    load_teacher_durations(L)
    segs, total = build_timeline(L)
    print(f"Durée : {total:.1f} s", flush=True)
    buf = build_audio(L, segs, total, args.audio)
    spec, level = spectrum(buf, total)
    state = dict(teach=dict(G_TEACH_DUR), L=L, segs=segs, total=total, spec=spec, level=level / max(1e-6, level.max()))
    G.update(state)

    if args.preview:
        for tp in args.preview:
            raw = render_frame(int(tp * FPS))
            Image.frombytes("RGB", (W, H), raw).save(f"preview_{tp:06.2f}.png")
            print("preview", tp, flush=True)
        return

    wav = args.out + ".wav"
    import wave
    with wave.open(wav, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((buf * 32767).astype("<i2").tobytes())
    nf = int(total * FPS)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", args.out], stdin=subprocess.PIPE)
    with Pool(args.jobs, initializer=init_worker, initargs=(state,)) as pool:
        for k, raw in enumerate(pool.imap(render_frame, range(nf), chunksize=6)):
            ff.stdin.write(raw)
            if k % 150 == 0:
                print(f"  image {k}/{nf}", flush=True)
    ff.stdin.close()
    ff.wait()
    os.remove(wav)
    print("OK ->", args.out)


if __name__ == "__main__":
    main()
