#!/usr/bin/env python3
"""Synthetic regressions; never opens or changes the user's library or app."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

import build_reader as builder
import build_from_ir
from epub_tools import EpubAuditError, audit_epub, segment_generated_epub
from math_rendering import MATH_NS, MathRenderingError, render_mathml, validate_mathml
from persist_epub import check_review, persist

M = "{" + MATH_NS + "}"


class MathTests(unittest.TestCase):
    def test_explicit_upright_letter_stays_distinct_from_italic_variable(self):
        root=ET.fromstring(render_mathml(r'\mathrm{E}+E'))
        letters=root.findall('.//'+M+'mi')
        self.assertEqual([node.text for node in letters],['E','E'])
        self.assertEqual(letters[0].get('mathvariant'),'normal')
        self.assertNotEqual(letters[1].get('mathvariant'),'normal')

    def test_aligned_rows_and_optional_spacing_are_structured(self):
        root = ET.fromstring(render_mathml(r"\begin{aligned}x&=1\\[2pt]y&=2\end{aligned}", True))
        self.assertEqual(len(root.findall('.//' + M + 'mtr')), 2)
        self.assertNotIn("2pt", ''.join(root.itertext()))
        self.assertTrue(all('text-align:' in node.get('style', '') for node in root.findall('.//' + M + 'mtd')))

    def test_unknown_command_is_not_silent_success(self):
        with self.assertRaises(MathRenderingError):
            render_mathml(r"x+\unknownmacro{y}")

    def test_nonletter_backslash_leak_is_rejected(self):
        with self.assertRaises(MathRenderingError):
            validate_mathml('<math xmlns="'+MATH_NS+'"><mi>\\2pt]</mi></math>')

    def test_original_tex_annotation_is_not_visible_fallback(self):
        validate_mathml('<math xmlns="'+MATH_NS+'"><semantics><mi>x</mi><annotation encoding="application/x-tex">\\alpha</annotation></semantics></math>')

    def test_sum_rule_remains_a_rule_not_a_fraction(self):
        root = ET.fromstring(render_mathml(r"\begin{array}{rr}a&b\\c&d\\\hline e&f\end{array}", True))
        rows = root.findall('.//' + M + 'mtr')
        self.assertFalse(root.findall('.//' + M + 'mfrac'))
        self.assertTrue(all('border-top:' in cell.get('style', '') for cell in rows[-1]))
        self.assertTrue(all('border-top:' not in cell.get('style', '') for cell in rows[0]))

    def test_limits_and_function_base_keep_correct_structure(self):
        root = ET.fromstring(render_mathml(r"\sum_{i=0}^{n}i+\log_2 n", True))
        self.assertEqual(len(root.findall('.//' + M + 'munderover')), 1)
        log = next(node for node in root.findall('.//' + M + 'msub') if ''.join(node[0].itertext()) == 'log')
        self.assertEqual(len(log), 2)
        self.assertEqual(''.join(log[1].itertext()), '2')

    def test_floor_ceiling_and_punctuation_are_not_rewritten(self):
        root = ET.fromstring(render_mathml(r"\lfloor x\rfloor\le\lceil x\rceil,"))
        text = ''.join(root.itertext())
        self.assertTrue(all(c in text for c in '⌊⌋⌈⌉,'))

    def test_fenced_tex_remains_literal_source(self):
        warnings = []
        fragment = builder.render_markdown("```tex\n\\unknownmacro{x}\n```\n", warnings)
        self.assertIn('unknownmacro', fragment)
        self.assertNotIn('math-source', fragment)
        self.assertEqual(warnings, [])


class PipelineTests(unittest.TestCase):
    def test_algorithm_preserves_step_order_comments_and_case_extent(self):
        block={'id':'a','type':'algorithm','sources':['u'],'source_reviewed':True,
               'source_evidence':[{'page':1,'observation':'Synthetic three-step case and trailing step.'}],
               'lines':[{'role':'title','runs':[{'text':'FIXUP'}]},
                        {'number':1,'indent':0,'runs':[{'text':'if x'},{'text':' // guard'}],'comment_start':1},
                        {'number':2,'indent':1,'runs':[{'text':'update y'}]},
                        {'number':3,'indent':1,'runs':[{'text':'update z'}]},
                        {'number':4,'indent':0,'runs':[{'text':'return'}]}],
               'case_groups':[{'start_row':1,'end_row':3,'latex':r'\}\ \text{case 1}'}]}
        renderer=build_from_ir.SemanticRenderer(Path('unused'),{}, {'u':{'kind':'text','text':'synthetic'}})
        dom=ET.fromstring(renderer.block(block))
        group=dom.find("./tbody[@class='algorithm-case-group']")
        self.assertEqual([tr.find('td').text for tr in group.findall('tr')],['1','2','3'])
        self.assertEqual(group.find(".//td[@class='algorithm-case']").get('rowspan'),'3')
        self.assertEqual(group.find(".//span[@class='algorithm-comment']").text,' // guard')
        self.assertEqual(dom.findall('./tbody')[-1].find('tr/td').text,'4')
        self.assertEqual(renderer.pending,[])
        block['case_groups'].append({'start_row':3,'end_row':4,'latex':r'\}\ \text{case 2}'})
        with self.assertRaisesRegex(ValueError,'overlapping'):
            build_from_ir.SemanticRenderer(Path('unused'),{}, {'u':{}}).block(block)

    def test_table_repeats_only_real_header_and_preserves_blank_corner(self):
        block={'id':'t','type':'table','sources':['u'],'header_rows':1,
               'source_reviewed':True,'source_evidence':[{'page':1,'observation':'Synthetic blank corner and two columns.'}],
               'rows':[[{'runs':[{'text':''}]},{'runs':[{'text':'cost'}]}],
                       [{'runs':[{'text':'step'}]},{'runs':[{'math':'n'}]}]]}
        renderer=build_from_ir.SemanticRenderer(Path('unused'),{}, {'u':{}})
        dom=ET.fromstring(renderer.block(block))
        self.assertEqual(len(dom.findall('./thead/tr')),1)
        self.assertEqual(len(dom.findall('./thead/tr/th')),2)
        self.assertEqual(''.join(dom.find('./thead/tr/th').itertext()),'')
        self.assertEqual(len(dom.findall('./tbody/tr')),1)
        self.assertEqual(dom.find('./tbody/tr/td').text,'step')

    def test_source_navigation_keeps_nonheading_target_hierarchy_and_authors(self):
        path=self.root/'source-navigation.epub'
        builder.make_epub(path,'Title','Fallback','<p id="copyright">Copyright.</p><h1 id="chapter">Chapter</h1><h2 id="section">Section</h2><p>Body.</p>',
                          'navigation-test',authors=['First author','Second author'],navigation=[
                              {'level':1,'target':'copyright','label':'Copyright'},
                              {'level':1,'target':'chapter','label':'Chapter'},
                              {'level':2,'target':'section','label':'Section'}])
        audit=audit_epub(path);self.assertEqual(audit['errors'],[])
        self.assertEqual(audit['authors'],['First author','Second author'])
        with zipfile.ZipFile(path) as archive:
            nav=ET.fromstring(archive.read('OEBPS/nav.xhtml'))
            content=ET.fromstring(archive.read('OEBPS/content.xhtml'))
        x='{http://www.w3.org/1999/xhtml}'
        roots=nav.findall('.//'+x+'nav/'+x+'ol/'+x+'li')
        self.assertEqual(len(roots),2)
        self.assertEqual(roots[0].find(x+'a').get('href'),'content.xhtml#copyright')
        self.assertEqual(roots[1].find(x+'ol/'+x+'li/'+x+'a').text,'Section')
        self.assertEqual(content.find('.//'+x+'p[@id="copyright"]').text,'Copyright.')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='comfortable-reader-pipeline-',dir=os.environ.get('COMFORTABLE_READER_TEST_DIR'))
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make(self, fragment='<h1 id="first">Start</h1><p>Text.</p>', source=b'Original\r\n', budget=0):
        path = self.root/'book.epub'
        builder.make_epub(path, 'Synthetic fixture', 'Test author', fragment, 'fixture-key',
                          source_archive_name='source.txt', original_source_bytes=source,
                          max_section_chars=budget)
        return path

    def test_real_input_bytes_not_reencoded_utf8_are_embedded(self):
        raw='# 标题\r\n\r\nA paragraph.\r\n'.encode('utf-16')
        source=self.root/'source.md';source.write_bytes(raw)
        with contextlib.redirect_stdout(io.StringIO()):
            code=builder.main([str(source),'--no-html','--epub',str(self.root/'bytes.epub')])
        self.assertEqual(code,0)
        with zipfile.ZipFile(self.root/'bytes.epub') as z:
            self.assertEqual(z.read('OEBPS/original/source.md'),raw)

    def test_segmentation_preserves_blocks_and_cross_links(self):
        math=render_mathml(r'\frac{x_1}{2}',True)
        fragment='<h1 id="a">A</h1><p>'+('alpha '*70)+'<a href="#z">end</a></p>'
        fragment+='<figure id="f"><img src="data:image/png;base64,AA=="/><figcaption>One whole caption.</figcaption></figure>'
        fragment+='<h2 id="b">B</h2><div class="math-block">'+math+'</div><p>'+('beta '*70)+'</p>'
        fragment+='<h2 id="z">Z</h2><p><a href="#a">back</a></p>'
        path=self.make(fragment,b'unchanged original',300)
        audit=audit_epub(path)
        self.assertEqual(audit['errors'],[])
        self.assertGreater(audit['spine_sections'],1)
        self.assertEqual(audit['mathml_formulas'],1)
        with zipfile.ZipFile(path) as z:
            body=[z.read(n).decode() for n in z.namelist() if n.endswith('.xhtml') and '/content' in n]
            self.assertEqual(sum('<figure' in text for text in body),1)
            self.assertEqual(sum('One whole caption.' in text for text in body),1)
            self.assertTrue(any('<math xmlns=' in text for text in body))
            self.assertEqual(z.read('OEBPS/original/source.txt'),b'unchanged original')

    def test_final_segmentation_is_audited_after_spine_rewrite(self):
        fragment='<h1 id="a">A</h1><p>'+('alpha '*70)+'</p><h2 id="b">B</h2><p>'+('beta '*70)+'</p>'
        path=self.make(fragment,b'original',120)
        audit=audit_epub(path)
        self.assertEqual(audit['errors'],[])
        self.assertGreater(audit['spine_sections'],1)
        with zipfile.ZipFile(path) as z:
            self.assertEqual(z.read('OEBPS/original/source.txt'),b'original')

    def test_build_from_ir_rejects_unmapped_full_scope(self):
        packet={'schema_version':1,'source':{'path':str(self.root/'source.pdf'),'sha256':'x','pages':1},'pages':[],'assets':{},'errors':[],'inventory_sha256':'x'}
        (self.root/'source.pdf').write_bytes(b'pdf')
        (self.root/'packet.json').write_text(json.dumps(packet),encoding='utf-8')
        ir={'packet':'packet.json','scope':{'kind':'full'},'title':'T','authors':['A'],'blocks':[],'issues':[]}
        (self.root/'ir.json').write_text(json.dumps(ir),encoding='utf-8')
        with self.assertRaises(Exception):
            build_from_ir.build(self.root/'ir.json',self.root/'out.epub')

    def test_broken_resource_blocks_audit(self):
        path=self.make('<p><img src="missing.png"/></p>')
        self.assertTrue(audit_epub(path)['errors'])

    def test_raw_pdf_cannot_be_one_step_imported(self):
        path=self.root/'input.pdf';path.write_bytes(b'%PDF-pretend fixture')
        with patch.object(builder,'add_to_calibre_library') as add,contextlib.redirect_stderr(io.StringIO()):
            result=builder.main([str(path),'--no-html','--add-to-library'])
        self.assertEqual(result,1)
        add.assert_not_called()

    def test_unknown_math_cannot_be_imported(self):
        path=self.root/'input.md';path.write_text(r'\[x+\unknownmacro{y}\]',encoding='utf-8')
        with patch.object(builder,'add_to_calibre_library') as add,contextlib.redirect_stderr(io.StringIO()):
            result=builder.main([str(path),'--no-html','--add-to-library'])
        self.assertEqual(result,1)
        add.assert_not_called()

    def test_missing_library_is_not_created(self):
        folder=self.root/'typo-library'
        with self.assertRaises(builder.BuildError):
            builder.add_to_calibre_library(self.make(),folder,'T','A','en','key')
        self.assertFalse(folder.exists())

    def review(self,path):
        report=self.root/'review.json'
        report.write_text(json.dumps({'epub_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                     'source_review':{'status':'passed','method':'verbatim_source','byte_equal':True,'source_sha256':'synthetic','evidence':[{'kind':'bytes','location':'original/source.txt','result':'equal'}]},
                                     'render_review':{'status':'passed','evidence':[{'kind':'visual','location':'synthetic fixture','result':'checked'}]},
                                     'unresolved_issues':[]}),encoding='utf-8')
        return report

    def test_stale_review_is_rejected(self):
        path=self.make();review=self.review(path)
        with self.assertRaises(EpubAuditError):
            check_review(review,'0'*64)

    def test_unresolved_review_is_rejected(self):
        path=self.make();review=self.review(path);data=json.loads(review.read_text());data['unresolved_issues']=['Wrong exponent'];review.write_text(json.dumps(data))
        with self.assertRaises(EpubAuditError):
            check_review(review,hashlib.sha256(path.read_bytes()).hexdigest())

    def test_persist_copies_exact_file_and_checks_repeat_identity(self):
        path=self.make();review=self.review(path);library=self.root/'library';library.mkdir()
        with contextlib.closing(sqlite3.connect(library/'metadata.db')) as db:
            db.executescript("CREATE TABLE books(id INTEGER,path TEXT,title TEXT);CREATE TABLE data(book INTEGER,name TEXT,format TEXT);INSERT INTO books VALUES(1,'Author/Book','Synthetic fixture');INSERT INTO data VALUES(1,'Book','EPUB');")
        calls=[]
        def fake_add(epub,folder,*args):
            status='already_present' if calls else 'added';calls.append(args[-1])
            stored=folder/'Author/Book/Book.epub';stored.parent.mkdir(parents=True,exist_ok=True)
            if status=='added':shutil.copyfile(epub,stored)
            return {'status':status,'book_ids':[1]}
        with patch('persist_epub.add_to_calibre_library',side_effect=fake_add):
            first=persist(path,library,review);second=persist(path,library,review)
        self.assertEqual(first['library']['status'],'added')
        self.assertEqual(second['library']['status'],'already_present')
        self.assertEqual(len(calls),1)
        self.assertEqual((library/'Author/Book/Book.epub').read_bytes(),path.read_bytes())
        self.assertEqual(first['delivery_status'],'awaiting_desktop_verification')


if __name__=='__main__':
    unittest.main()
