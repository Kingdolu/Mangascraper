# MangaScraper

MangaPark Scraper 🐾

A Python-based MangaPark scraper with a GUI for easy manga chapter downloads. Originally terminal-based, now upgraded with a Tkinter interface to select chapters and monitor download progress.

⚡ Features

GUI Interface using Tkinter:

Input field for the MangaPark URL.

Chapter list for single or multiple selection.

Input field to download a range of chapters (supports fractional chapters like 34.5).

Download buttons:

Download Selected – downloads highlighted chapters.

Download Range – downloads all chapters in the entered range.

Progress bar showing real-time download status.

PDF Export: Downloads chapters as PDF files.

Organized folders: Saves downloads to ./downloads/<MANGA_TITLE>/.

Async image downloading for faster downloads.

Handles chapters with fractional numbers correctly.

Optional debug logging for troubleshooting.

🛠 Requirements

Python 3.11+

Install dependencies:

pip install requests beautifulsoup4 playwright img2pdf aiohttp tqdm pyyaml pillow
playwright install  # required if using Playwright fallback

🚀 Usage
Terminal Mode
python mangapark_scraper.py "https://mangapark.io/title/127529-en-the-lazy-lord-masters-the-sword"


Optional arguments:

-o, --output : Output directory (default: current folder)

-c, --chapters : Chapter selection (e.g., 1-3, 5, Black Wolf, all)

--no-async : Disable async downloads

--debug : Enable debug logging

GUI Mode

Run:

python mangapark_gui.py


Steps:

Paste Manga URL in the URL field.

Click Fetch Chapters to load all chapters.

Option 1 – Download Selected Chapters

Highlight chapters in the list.

Click Download Selected.

Option 2 – Download Range

Enter a chapter range (e.g., 1-5 or 12.5-14) in the range field.

Click Download Range.

Monitor progress bar and status label for updates.

Downloads are saved in ./downloads/<MANGA_TITLE>/.

📌 Notes

Fractional chapters (e.g., 34.5) are supported in both GUI and terminal.

PDFs are named like Chapter_001.pdf, Chapter_002.pdf, etc., with zero-padding.

GUI uses threading to keep the interface responsive during downloads.

You can use selected chapters and range download simultaneously.

📝 License

MIT License — free to use and modify.se and modify.r-title folder.

Metadata saving is disabled by default
