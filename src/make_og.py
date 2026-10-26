"""Draw site/og.png (1200x630 link-preview card) and site/favicon.svg in the site's style.

    python src/make_og.py   (rerun after the backtest numbers change)
"""

from PIL import Image, ImageDraw, ImageFont

from common import OUTPUTS, ROOT

BG, PANEL, INK, MUTED, BRASS, BRASS_DIM = "#11151b", "#181e26", "#ece6d8", "#9aa2ab", "#c8a45d", "#6b5832"
FONTS = "C:/Windows/Fonts/"


def font(name, size):
    try:
        return ImageFont.truetype(FONTS + name, size)
    except OSError:
        return ImageFont.load_default()


def main():
    import pandas as pd
    ms = pd.read_csv(OUTPUTS / "dashboard" / "model_scores.csv")
    ms = ms[ms["phase"] == "test"].set_index("model")
    ens, veg, elo = (ms.loc[k].to_dict() for k in ("Ensemble (5 models)", "Vegas moneyline", "Elo rating"))

    im = Image.new("RGB", (1200, 630), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1200, 8], fill=BRASS)
    # seal
    cx, cy, r = 150, 200, 78
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=BRASS, width=3)
    d.ellipse([cx - r + 11, cy - r + 11, cx + r - 11, cy + r - 11], outline=BRASS_DIM, width=2)
    seal = font("georgia.ttf", 46)
    d.text((cx, cy), "K&A", font=seal, fill=BRASS, anchor="mm")
    d.text((270, 158), "Krish & Associates", font=font("georgia.ttf", 72), fill=INK, anchor="lm")
    d.text((274, 232), "NFL WIN PROBABILITY  ·  COUNSEL BY NEURAL NETWORK", font=font("segoeui.ttf", 22), fill=MUTED, anchor="lm")
    # results strip
    d.rectangle([70, 340, 1130, 540], fill=PANEL, outline="#2b333d")
    cols = [(f"{ens['accuracy'] * 100:.1f}%", "winners picked on 1,218\nheld-out games"),
            (f"{veg['accuracy'] * 100:.1f}%", "Vegas on the\nsame games"),
            (f"+{(ens['accuracy'] - elo['accuracy']) * 100:.1f} pts", "over a classic\nElo rating")]
    for i, (big, lab) in enumerate(cols):
        x = 110 + i * 350
        d.text((x, 395), big, font=font("georgia.ttf", 64), fill=BRASS if i == 0 else INK, anchor="lm")
        d.multiline_text((x, 470), lab, font=font("segoeui.ttf", 22), fill=MUTED, spacing=4)
    d.text((70, 585), "Neural-network ensemble · PyTorch · walk-forward validated · weekly live picks",
           font=font("segoeui.ttf", 22), fill=MUTED, anchor="lm")
    im.save(ROOT / "site" / "og.png", optimize=True)

    (ROOT / "site" / "favicon.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        f'<rect width="64" height="64" rx="12" fill="{BG}"/>'
        f'<circle cx="32" cy="32" r="27" fill="none" stroke="{BRASS}" stroke-width="2.5"/>'
        f'<text x="32" y="40" text-anchor="middle" font-family="Georgia,serif" font-size="21" fill="{BRASS}">K&amp;A</text>'
        '</svg>')
    print("wrote site/og.png and site/favicon.svg")


if __name__ == "__main__":
    main()
