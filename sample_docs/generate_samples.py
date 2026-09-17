#!/usr/bin/env python3
"""Generate the two small sample PDFs bundled with this project.

Original content, written for this project — not copied from anywhere.
Run this script to regenerate sample_docs/*.pdf if they are ever deleted:

    python sample_docs/generate_samples.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

HERE = Path(__file__).parent


def write_pdf(path: Path, pages: list[list[str]]) -> None:
    """Write a multi-page PDF, where `pages` is a list of pages, each a list
    of paragraph strings for that page."""
    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER
    for page_paragraphs in pages:
        y = height - 1 * inch
        c.setFont("Helvetica", 11)
        for paragraph in page_paragraphs:
            for line in _wrap(paragraph, 90):
                c.drawString(1 * inch, y, line)
                y -= 16
            y -= 10
        c.showPage()
    c.save()


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for w in words:
        candidate = (current + " " + w).strip()
        if len(candidate) > width:
            lines.append(current)
            current = w
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


HYDROPONICS_PAGES = [
    [
        "Backyard Hydroponics: A Practical Guide",
        "Hydroponics is a method of growing plants without soil, in which the "
        "roots sit directly in a nutrient-rich water solution. Because the "
        "plant does not have to search through soil for nutrients, growth "
        "rates in a well-tuned hydroponic system are often faster than in "
        "traditional garden beds.",
        "The main conclusion of this guide is that a simple deep water "
        "culture setup, built from a plastic tote and an aquarium air pump, "
        "is the cheapest reliable way for a beginner to start growing leafy "
        "greens indoors year round.",
    ],
    [
        "Choosing a System",
        "There are several common hydroponic system types: deep water "
        "culture, nutrient film technique, ebb and flow, and drip systems. "
        "Deep water culture is the easiest to build and maintain for a "
        "first-time grower because it has no moving parts other than an air "
        "pump.",
        "Nutrient film technique circulates a thin film of nutrient solution "
        "over the roots continuously and is popular in commercial lettuce "
        "farms, but it requires a reliable pump and is less forgiving of "
        "power outages than deep water culture.",
    ],
    [
        "Nutrients and pH",
        "Plants grown hydroponically need a balanced nutrient solution "
        "containing nitrogen, phosphorus, potassium, and trace minerals such "
        "as iron and magnesium. Most hobbyists use a commercial two-part "
        "nutrient concentrate mixed into water at the ratio printed on the "
        "bottle.",
        "The pH of the nutrient solution should be kept between 5.5 and 6.5 "
        "for most leafy greens, since nutrients become harder for roots to "
        "absorb outside that range even if the nutrients themselves are "
        "present in the water.",
    ],
    [
        "Common Problems",
        "Root rot is the most common failure in hydroponic systems and is "
        "usually caused by warm, poorly oxygenated water. Keeping the "
        "reservoir below 21 degrees Celsius and running the air pump "
        "continuously prevents most cases.",
        "Algae growth on sunlit reservoir walls competes with plants for "
        "nutrients and can be avoided by using an opaque container or "
        "wrapping a clear tote in dark plastic sheeting.",
    ],
]

LIGHTHOUSE_PAGES = [
    [
        "A Short History of Coastal Lighthouses",
        "Lighthouses have guided sailors along dangerous coastlines for "
        "thousands of years, evolving from simple bonfires on hilltops to "
        "the automated, solar-powered beacons used today.",
        "The earliest known lighthouse, the Pharos of Alexandria, was built "
        "around 280 BC and is believed to have stood over 100 meters tall, "
        "making it one of the tallest structures in the ancient world.",
    ],
    [
        "The Age of Stone Towers",
        "During the eighteenth and nineteenth centuries, engineers such as "
        "John Smeaton pioneered the use of interlocking stone blocks to "
        "build lighthouses that could survive direct waves crashing over "
        "them during storms.",
        "The Eddystone Lighthouse off the coast of Cornwall was rebuilt "
        "several times, and Smeaton's 1759 design became the template for "
        "wave-swept lighthouse construction for the next century.",
    ],
    [
        "Automation and Decline of Keepers",
        "Through the twentieth century, electric lamps, timers, and "
        "eventually solar panels with photocells gradually replaced the "
        "human lighthouse keepers who had once lived on site to trim wicks "
        "and wind clockwork rotation mechanisms by hand.",
        "By the 1990s most lighthouses in wealthy coastal nations had been "
        "fully automated, and the keeper became a largely historical "
        "profession, with many former keeper's quarters converted into "
        "small museums.",
    ],
]


def main() -> None:
    write_pdf(HERE / "hydroponics_guide.pdf", HYDROPONICS_PAGES)
    write_pdf(HERE / "lighthouse_history.pdf", LIGHTHOUSE_PAGES)
    print("Wrote sample_docs/hydroponics_guide.pdf and sample_docs/lighthouse_history.pdf")


if __name__ == "__main__":
    main()
