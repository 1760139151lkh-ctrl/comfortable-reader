"""Focused regressions for shared-reader provenance and EPUB compatibility."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'skill' / 'comfortable-reader' / 'scripts'))
import build_reader
import preview_existing_book
from reader_build_receipt import ReaderBuildError, verify_reader_dist
from streamed_epub import prepare_reader


def sample_epub(path: Path) -> None:
    container = b'''<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OPS/package.opf" media-type="application/oebps-package+xml"/></rootfiles></container>'''
    opf = b'''<?xml version="1.0" encoding="utf-8"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="id">urn:uuid:00000000-0000-4000-8000-000000000123</dc:identifier><dc:title>Another Book</dc:title><dc:creator>Example Author</dc:creator><dc:language>en</dc:language><meta property="dcterms:modified">2024-05-06T00:00:00Z</meta></metadata><manifest><item id="chapter" href="Text/one.xhtml" media-type="application/xhtml+xml"/><item id="nav" href="Navigation/toc.xhtml" media-type="application/xhtml+xml" properties="nav"/></manifest><spine><itemref idref="chapter"/></spine></package>'''
    chapter = b'''<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml"><head><title>One</title></head><body><h1>One</h1><p>Actual chapter text in a non-OEBPS package.</p></body></html>'''
    nav = b'''<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml"><head><title>Contents</title></head><body><nav><a href="../Text/one.xhtml">One</a></nav></body></html>'''
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('mimetype', b'application/epub+zip', compress_type=zipfile.ZIP_STORED)
        archive.writestr('META-INF/container.xml', container)
        archive.writestr('OPS/package.opf', opf)
        archive.writestr('OPS/Text/one.xhtml', chapter)
        archive.writestr('OPS/Navigation/toc.xhtml', nav)


class ReaderBuildContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='reader-contract-', dir=TESTS)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(TESTS.resolve()))

    def tearDown(self):
        self.assertTrue(self.root.is_relative_to(TESTS.resolve()))
        self.temporary.cleanup()

    def test_unreceipted_index_is_not_a_shared_reader_build(self):
        fake = self.root / 'fake-shell'
        fake.mkdir()
        (fake / 'index.html').write_text('<title>舒适阅读书库</title><main id="app"></main>', encoding='utf-8')
        with self.assertRaisesRegex(ReaderBuildError, '缺少当前构建的来源收据'):
            verify_reader_dist(fake)

    def test_non_oebps_package_maps_spine_and_keeps_original_epub_bytes(self):
        epub = self.root / 'original.epub'
        sample_epub(epub)
        destination = self.root / 'streamed'
        reader, mapping, _ = prepare_reader(epub, destination,
                                             {'chapters': [{'id': 'c01', 'href': 'Text/one.xhtml'}]})
        self.assertEqual(reader['package']['path'], 'reader/OPS/package.opf')
        self.assertEqual(mapping['c01']['path'], 'reader/OPS/Text/one.xhtml')
        self.assertIn('reader/OPS/Navigation/toc.xhtml', [row['path'] for row in reader['support']])

        pack = {'book_revision_sha256': hashlib.sha256(epub.read_bytes()).hexdigest(),
                'book_uuid': 'urn:uuid:00000000-0000-4000-8000-000000000123',
                'chapters': [{'id': 'c01', 'href': 'Text/one.xhtml', 'title': 'One'}],
                'assets': [], 'activities': [], 'source_claims': []}
        pack_file = self.root / 'learning.json'
        pack_file.write_text(json.dumps(pack), encoding='utf-8')
        shell = self.root / 'shell'
        shell.mkdir()
        (shell / 'index.html').write_text('<main id="app"></main>', encoding='utf-8')
        output = self.root / 'preview'
        argv = ['preview_existing_book.py', '--epub', str(epub), '--learning-pack', str(pack_file),
                '--content-root', str(self.root), '--shell', str(shell), '--out', str(output),
                '--revision', '0.1.0']
        receipt = {'sourceTreeSha256': 'test-only-source', 'buildTreeSha256': 'test-only-build'}
        with patch.object(sys, 'argv', argv), patch.object(preview_existing_book, 'verify_reader_dist', return_value=receipt), patch('sys.stdout', new_callable=io.StringIO):
            preview_existing_book.main()
        book = json.loads((output / 'books' / 'existing-book' / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(book['title'], 'Another Book')
        self.assertEqual(book['author'], 'Example Author')
        self.assertEqual(book['language'], 'en')
        self.assertEqual(book['revisionModified'], '2024-05-06T00:00:00Z')
        self.assertIn('1 个已登记章节', book['description'])
        self.assertNotIn('47 章', book['description'])
        self.assertEqual(book['chapters'][0]['readingDocument']['path'], 'reader/OPS/Text/one.xhtml')
        self.assertEqual(epub.read_bytes(), (output / 'ebooks' / 'existing-book-0.1.0.epub').read_bytes())

    def test_open_requires_an_explicit_compatibility_preview_target(self):
        source = self.root / 'sample.txt'
        source.write_text('A small readable paragraph.', encoding='utf-8')
        epub = self.root / 'draft.epub'
        with patch('sys.stdout', new_callable=io.StringIO), patch('sys.stderr', new_callable=io.StringIO) as stderr, patch.object(build_reader, 'open_artifact') as open_artifact:
            result = build_reader.main([str(source), '--no-html', '--epub', str(epub), '--open'])
            self.assertEqual(result, 1)
            self.assertIn('明确指定 --reader', stderr.getvalue())
            open_artifact.assert_not_called()
        self.assertFalse(epub.exists())

    def test_explicit_open_reports_compatibility_scope_without_desktop_verification(self):
        source = self.root / 'sample.txt'
        source.write_text('A small readable paragraph.', encoding='utf-8')
        html = self.root / 'preview.html'
        with patch('sys.stdout', new_callable=io.StringIO) as stdout, patch.object(build_reader, 'open_artifact', return_value=('html', 'test-browser')) as open_artifact:
            result = build_reader.main([str(source), '--no-epub', '--output', str(html), '--open', '--reader', 'html'])
            self.assertEqual(result, 0)
            self.assertEqual(open_artifact.call_count, 1)
            reported = json.loads(stdout.getvalue())
        self.assertEqual(reported['opened']['reader'], 'html')
        self.assertEqual(reported['opened']['scope'], 'temporary_compatibility_preview')
        self.assertIs(reported['opened']['desktop_verified'], False)
        self.assertEqual(reported['delivery_status'], 'draft_not_imported')


if __name__ == '__main__':
    unittest.main()
