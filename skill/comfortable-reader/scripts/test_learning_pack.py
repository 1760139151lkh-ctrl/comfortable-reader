"""Behavioral fixtures: no real library writes, no execution or trust grants."""
import json,tempfile,unittest,zipfile,wave
from pathlib import Path
import learning_pack as tool

class PackTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        (self.root/'example.py').write_text('print(2 + 3)\n',encoding='utf-8')
        self.epub=self.root/'book.epub'
        with zipfile.ZipFile(self.epub,'w') as z:
            z.writestr('META-INF/container.xml','<container><rootfiles><rootfile full-path="content.opf"/></rootfiles></container>')
            z.writestr('content.opf','<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata><dc:identifier>urn:uuid:test-book</dc:identifier></metadata><manifest><item id="c" href="chapter.xhtml"/></manifest><spine><itemref idref="c"/></spine></package>')
            z.writestr('chapter.xhtml','<html><body><p>One calculation.</p></body></html>')
        self.spec={'package_id':'example','revision':'1','chapters':[{'id':'one','href':'chapter.xhtml','assets':['program']}],'assets':[{'id':'program','relative_path':'example.py','kind':'code','title':'Compute a sum'}],'href_assets':{'code.xhtml':'program'},'activities':[{'id':'sum','chapter':'one','entry_asset':'program','runtime':'local-python'}],'source_claims':[]}
        self.pack=tool.build(self.spec,self.epub,self.root)
    def test_round_trip_preserves_source_and_grants_nothing(self):
        before=self.epub.read_bytes();report=tool.audit(self.pack,self.epub,self.root)
        self.assertEqual(before,self.epub.read_bytes());self.assertFalse(report['execution_authorized']);self.assertEqual(report['desktop_verification'],'awaiting_desktop_verification')
    def test_modified_code_rejected(self):
        (self.root/'example.py').write_text('print(500)\n')
        with self.assertRaisesRegex(ValueError,'asset bytes changed'):tool.audit(self.pack,self.epub,self.root)
    def test_missing_asset_rejected(self):
        (self.root/'example.py').unlink()
        with self.assertRaises(OSError):tool.audit(self.pack,self.epub,self.root)
    def test_stale_epub_rejected(self):
        with zipfile.ZipFile(self.epub,'a') as z:z.writestr('new.txt','another revision')
        with self.assertRaisesRegex(ValueError,'stale'):tool.audit(self.pack,self.epub,self.root)
    def test_book_cannot_supply_commands(self):
        self.pack['activities'][0]['command']='python arbitrary.py'
        with self.assertRaisesRegex(ValueError,'authority'):tool.audit(self.pack,self.epub,self.root)
    def test_path_escapes_rejected(self):
        for value in ('../secret','C:/secret','x:stream','%2e%2e/secret','a\\b','NUL.txt','file.','file '):
            with self.subTest(value=value),self.assertRaises(ValueError):tool.relative_file(self.root,value)
    def test_unknown_chapter_rejected(self):
        self.pack['chapters'][0]['href']='not-in-book.xhtml'
        with self.assertRaisesRegex(ValueError,'spine'):tool.audit(self.pack,self.epub,self.root)
    def test_unsafe_source_scheme_rejected(self):
        self.pack['source_claims']=[{'url':'file:///C:/secret'}]
        with self.assertRaisesRegex(ValueError,'URL'):tool.audit(self.pack,self.epub,self.root)
    def test_audio_keeps_declared_provenance_and_measures_actual_format(self):
        with wave.open(str(self.root/'sound.wav'),'wb') as audio:
            audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(8000);audio.writeframes(b'\0\0'*80)
        self.spec['assets'].append({'id':'sound','relative_path':'sound.wav','kind':'audio','title':'An example','media':{'identity':'Synthetic fixture, not field audio','sample_rate':999}})
        pack=tool.build(self.spec,self.epub,self.root)
        media=next(a for a in pack['assets'] if a['id']=='sound')['media']
        self.assertEqual(media['identity'],'Synthetic fixture, not field audio');self.assertEqual(media['sample_rate'],8000);self.assertEqual(media['frames'],80)

if __name__=='__main__':unittest.main()
