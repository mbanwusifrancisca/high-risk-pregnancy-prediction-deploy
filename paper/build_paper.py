#!/usr/bin/env python3
"""Merge all paper sections into a single Markdown file with image references,
then convert to .docx using pandoc."""

import subprocess
import pathlib

PAPER_DIR = pathlib.Path(__file__).parent
PROJECT_ROOT = PAPER_DIR.parent
FIGURES = PROJECT_ROOT / "reports" / "figures"

SECTIONS = [
    "ABSTRACT_AND_FRONT_MATTER.md",
    "CHAPTER_1_INTRODUCTION.md",
    "CHAPTER_2_LITERATURE_REVIEW.md",
    "CHAPTER_3_METHODOLOGY.md",
    "CHAPTER_4_RESULTS_AND_DISCUSSION.md",
    "CHAPTER_5_CONCLUSION.md",
    "REFERENCES.md",
]

# Figure insertion points — keyed by text that should PRECEDE the figure
FIGURE_INSERTIONS = {
    "The MLP's training history (Figure 4.1)": {
        "path": FIGURES / "mlp_training_history.png",
        "caption": "**Figure 4.1:** MLP Training History — Loss (left) and Accuracy (right) over epochs for training and validation sets.",
    },
    "Figure 4.2 presents the ROC curves": {
        "path": FIGURES / "roc_comparison.png",
        "caption": "**Figure 4.2:** ROC Curve Comparison — All Models. AUC values shown in legend.",
    },
    "The confusion matrices for each model reveal distinct classification patterns:": {
        "path_list": [
            (FIGURES / "cm_logistic_regression.png", "Logistic Regression"),
            (FIGURES / "cm_decision_tree.png", "Decision Tree"),
            (FIGURES / "cm_random_forest.png", "Random Forest"),
            (FIGURES / "cm_mlp.png", "MLP"),
        ],
        "caption": "**Figure 4.3:** Confusion Matrices — All Models.",
    },
    "Feature importance was analysed across three model types": {
        "path": FIGURES / "feature_importance_comparison.png",
        "caption": "**Figure 4.4:** Feature Importance Comparison (Top 10) across Logistic Regression, Decision Tree, and Random Forest.",
    },
}


def read_section(filename):
    path = PAPER_DIR / filename
    return path.read_text(encoding="utf-8")


def insert_figures(text):
    """Insert figure markdown after the paragraph containing the trigger text."""
    for trigger, fig_info in FIGURE_INSERTIONS.items():
        if trigger not in text:
            print(f"  ⚠ trigger not found: {trigger[:60]}...")
            continue

        # Build figure markdown
        if "path_list" in fig_info:
            imgs = "\n\n".join(
                f"![Confusion Matrix — {label}]({p})" for p, label in fig_info["path_list"]
            )
            fig_md = f"\n\n{imgs}\n\n{fig_info['caption']}\n"
        else:
            fig_md = f"\n\n![{fig_info['caption']}]({fig_info['path']})\n\n{fig_info['caption']}\n"

        # Find the end of the paragraph containing the trigger
        idx = text.index(trigger)
        # Find the next double-newline (paragraph break) after the trigger
        para_end = text.find("\n\n", idx)
        if para_end == -1:
            para_end = len(text)
        text = text[:para_end] + fig_md + text[para_end:]

    return text


def main():
    print("Merging paper sections...")
    merged = []
    for section in SECTIONS:
        print(f"  + {section}")
        content = read_section(section)
        merged.append(content.strip())

    full_text = "\n\n\\newpage\n\n".join(merged)

    print("Inserting figures...")
    full_text = insert_figures(full_text)

    # Write merged markdown
    merged_md = PAPER_DIR / "FULL_PAPER.md"
    merged_md.write_text(full_text, encoding="utf-8")
    print(f"Merged markdown: {merged_md}")

    # Convert to docx
    output_docx = PAPER_DIR / "High_Risk_Pregnancy_Prediction_Paper.docx"
    cmd = [
        "pandoc",
        str(merged_md),
        "-o", str(output_docx),
        "--from", "markdown",
        "--to", "docx",
        "--standalone",
        "--toc",
        "--toc-depth=3",
        "--resource-path", str(PROJECT_ROOT),
    ]
    print(f"Converting to docx...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Pandoc error:\n{result.stderr}")
    else:
        print(f"DOCX output: {output_docx}")

    print("Done!")


if __name__ == "__main__":
    main()
