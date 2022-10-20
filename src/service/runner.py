import logging

import typer

from typing import Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    handlers=[
                        logging.FileHandler("./logs/app.log", "a")
                    ],
                    format='[%(asctime)s | %(levelname)s]: %(message)s',
                    datefmt='%m.%d.%Y %H:%M:%S')


app = typer.Typer()

from luhn_summarizer import LuhnExtractiveSummarizer
summarizer = LuhnExtractiveSummarizer()


@app.command()
def summarize(document_path: Optional[Path]):
    """
    API call to model.summarize()
    :param document_path: Article Text for summarization
    """
    with open(document_path, "r", encoding="utf8") as f:
        doc = "".join(f.readlines())

    summary = summarizer.summarize(doc)

    # saving results
    save_path = Path("./data/summary_results/ru_summary_test.txt")
    with open(save_path, "w", encoding="utf8") as f:
        f.writelines(summary)

    logging.info(f"Summary for {document_path} has been saved to {save_path} successfully!")



if __name__ == '__main__':
    app()
