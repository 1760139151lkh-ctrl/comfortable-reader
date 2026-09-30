from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from book_format import BookError, validate_book
from new_book import epub_chapter


class AuthoringFlow(unittest.TestCase):
    def test_generated_plain_book_uses_same_validator(self):
        spec, documents = validate_book(ROOT / 'examples' / 'books' / 'observation-notes', 'observation-notes')
        self.assertEqual(len(documents), 2)
        self.assertEqual(spec['activities'], [])
        self.assertEqual(spec['rights']['status'], 'approved')

    def test_simple_epub_import_preserves_text_but_rejects_image_loss(self):
        text = b'<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>A</h1><p>one <strong>two</strong></p></body></html>'
        draft, title = epub_chapter(text, 'Fallback')
        self.assertEqual(title, 'A')
        self.assertIn('one **two**', draft)
        rich = b'<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>A</h1><img src="figure.png"/></body></html>'
        with self.assertRaisesRegex(BookError, 'img'):
            epub_chapter(rich, 'Fallback')


if __name__ == '__main__': unittest.main()
