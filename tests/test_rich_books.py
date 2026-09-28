from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from book_format import BookError, validate_book
from export_epub import export_book


class SemanticBookSource(unittest.TestCase):
    def test_same_source_exports_identical_bytes_and_preserves_math(self):
        source=ROOT/'books/mathematical-equivalence'
        spec,_=validate_book(source,source.name)
        with tempfile.TemporaryDirectory() as temporary:
            first=Path(temporary)/'a.epub';second=Path(temporary)/'b.epub'
            export_book(source,first);export_book(source,second)
            self.assertEqual(first.read_bytes(),second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                self.assertGreater(sum(archive.read(name).count(b'<math') for name in archive.namelist() if name.endswith('.xhtml')),500)
                for name in spec['epubSource']['files']:
                    self.assertEqual(archive.read(name),(source/'epub'/name).read_bytes())

    def test_untrusted_rich_source_cannot_add_script_or_remote_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'mathematical-equivalence'
            shutil.copytree(ROOT/'books/mathematical-equivalence',source)
            chapter=source/'epub/OEBPS/ch00.xhtml';original=chapter.read_text(encoding='utf-8')
            for payload in ('<script>alert(1)</script>','<img src="https://example.invalid/tracker.png"/>'):
                chapter.write_text(original.replace('</body>',payload+'</body>'),encoding='utf-8')
                with self.assertRaises(BookError):validate_book(source,source.name)

    def test_public_catalogue_keeps_examples_separate(self):
        listing=json.loads((ROOT/'books/publication-list.json').read_text(encoding='utf-8'))
        self.assertIn('ai-principles',listing['books'])
        for slug in listing['books']:
            book=json.loads((ROOT/'books'/slug/'book.json').read_text(encoding='utf-8'))
            self.assertFalse(book.get('example'))
            self.assertEqual(book['rights']['status'],'approved')
        self.assertNotIn('measurement-lab',listing['books'])


if __name__=='__main__':unittest.main()
