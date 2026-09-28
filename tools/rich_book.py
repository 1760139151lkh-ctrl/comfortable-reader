"""Build a reviewed semantic EPUB source without flattening its MathML or CSS.

This adapter only packages positively listed inert files. It never imports or
executes a book's Python, installs dependencies, or grants native permissions.
"""
from pathlib import Path
import hashlib
import json
import posixpath
import re
import uuid
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlsplit
from book_format import BookError, safe_relative, valid_id, VERSION, MODIFIED, file_identity


def xml_document(raw: bytes, label: str):
    if re.search(rb'<!ENTITY|<!DOCTYPE[^>]+(?:SYSTEM|PUBLIC)', raw, re.I):
        raise BookError(f'{label}: 不接受外部 DTD 或实体')
    try:
        return ET.fromstring(raw)
    except ET.ParseError as error:
        raise BookError(f'{label}: XML 不完整：{error}') from error


def validate_rich_book(root: Path, data: dict, slug: str):
    if data.get('schemaVersion') != 1 or data.get('slug') != valid_id(slug, '书名目录'):
        raise BookError('书籍身份与目录不一致')
    if data.get('id') != 'urn:uuid:' + str(uuid.UUID(data.get('id', '').removeprefix('urn:uuid:'))):
        raise BookError('书籍须有稳定 UUID')
    if not VERSION.fullmatch(data.get('revision', '')) or not MODIFIED.fullmatch(data.get('revisionModified', '')):
        raise BookError('书籍修订及日期无效')
    source = data.get('epubSource', {})
    names = source.get('files', [])
    if not names or len(names) > 15000 or len(names) != len(set(names)):
        raise BookError('EPUB 源文件清单为空、重复或过大')
    files = {name: safe_relative(root, source['root'] + '/' + name) for name in names}
    if 'META-INF/container.xml' not in files or 'mimetype' in files:
        raise BookError('EPUB 容器清单缺失；mimetype 由导出器产生')
    documents = {}
    total = 0
    for name, path in files.items():
        total += path.stat().st_size
        if path.stat().st_size > 16 * 1024 * 1024 or total > 128 * 1024 * 1024:
            raise BookError('EPUB 阅读源超过单件或全书预算；大材料应单独声明')
        if path.suffix not in ('.xml', '.opf', '.ncx', '.xhtml', '.css', '.png'):
            raise BookError(f'EPUB 源文件类型尚未适配：{name}')
        raw = path.read_bytes()
        if path.suffix == '.css' and re.search(rb'url\s*\(|@import|expression\s*\(', raw, re.I):
            raise BookError(f'{name}: 样式不能隐式读取网络材料')
        if path.suffix not in ('.xml', '.opf', '.ncx', '.xhtml'):
            continue
        tree = xml_document(raw, name)
        documents[name] = tree
        for element in tree.iter():
            tag = element.tag.rsplit('}', 1)[-1]
            if tag in ('script', 'iframe', 'object', 'embed', 'form', 'input', 'textarea', 'button', 'foreignObject'):
                raise BookError(f'{name}: 含不可在书页执行的元素 {tag}')
            for key, value in element.attrib.items():
                if key.lower().startswith('on') or key == 'srcset' or re.match(r'(javascript|vbscript|file):', value.strip(), re.I):
                    raise BookError(f'{name}: 含事件或不安全地址')
                if key == 'style' and re.search(r'url\s*\(|@import|expression\s*\(', value, re.I):
                    raise BookError(f'{name}: 行内样式含隐式资源')
            if tag == 'style' and re.search(r'url\s*\(|@import|expression\s*\(', element.text or '', re.I):
                raise BookError(f'{name}: 内嵌样式含隐式资源')
            href = element.get('src') if tag == 'img' else element.get('href')
            if not href:
                continue
            if href.startswith('data:image/png;base64,') and tag == 'img':
                continue
            address = urlsplit(href)
            if address.scheme:
                if tag != 'a' or address.scheme not in ('http', 'https', 'mailto'):
                    raise BookError(f'{name}: 阅读会隐式访问外部资源')
                continue
            target = posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(address.path))) if address.path else name
            if target not in files:
                raise BookError(f'{name}: 内部链接目标未纳入：{target}')
    container = documents['META-INF/container.xml']
    package_name = container.find('.//{*}rootfile').get('full-path')
    package = documents.get(package_name)
    if package is None:
        raise BookError('容器指向的 OPF 未纳入')
    identifier = package.find('.//{*}metadata/{*}identifier')
    if identifier is None or identifier.text != data['id']:
        raise BookError('正文与书籍声明的 UUID 不同')
    manifest = package.find('{*}manifest')
    items = {item.get('id'): posixpath.normpath(posixpath.join(posixpath.dirname(package_name), unquote(item.get('href')))) for item in manifest}
    if any(name not in files for name in items.values()):
        raise BookError('EPUB 清单引用未纳入文件')
    spine = package.find('{*}spine')
    if any(item.get('idref') not in items for item in spine):
        raise BookError('EPUB 阅读顺序引用未知文件')
    chapter_ids = set()
    result = {}
    for chapter in data['chapters']:
        key = valid_id(chapter['id'], '章 ID')
        if key in chapter_ids:
            raise BookError('章节 ID 重复')
        chapter_ids.add(key)
        name = 'OEBPS/' + chapter['href']
        if name not in documents:
            raise BookError(f'{key}: 正文章节未纳入')
        result[key] = {'sourceFormat': 'epub-source@1', 'references': []}
    resource_ids = set()
    for resource in data.get('resources', []):
        key = valid_id(resource['id'], '材料 ID')
        if key in resource_ids or resource.get('kind') not in ('code', 'data', 'image', 'audio', 'video', 'pdf', 'model', 'geometry', 'text'):
            raise BookError('材料 ID 重复或类型无效')
        resource_ids.add(key)
        path = safe_relative(root, resource['path'])
        if file_identity(path) != {k: resource[k] for k in ('bytes', 'sha256')}:
            raise BookError(f'{key}: 材料已变化，请核对并更新书籍源声明')
        if path.stat().st_size > 128 * 1024 * 1024:
            raise BookError(f'{key}: 材料超过当前单件预算')
        if not set(resource.get('chapters', [])).issubset(chapter_ids):
            raise BookError(f'{key}: 所属章节不存在')
        if resource.get('kind') == 'image':
            header = path.read_bytes()[:24]
            if not header.startswith(b'\x89PNG\r\n\x1a\n'):
                raise BookError(f'{key}: 此源接口的插图需 PNG')
    for activity in data.get('activities', []):
        valid_id(activity['id'], '活动 ID')
        if activity['chapter'] not in chapter_ids or not activity['required'] or not set(activity['required'] + activity['optional']).issubset(resource_ids):
            raise BookError('活动章节或必需材料不存在')
        if activity.get('capability') != 'native-python@1' or any(k in activity for k in ('trusted', 'shell', 'command', 'cwd', 'env', 'permissions')):
            raise BookError('活动不能携带本机运行权限')
    if data.get('studySource'):
        study = json.loads(safe_relative(root, data['studySource']).read_text(encoding='utf-8'))
        if study.get('book_uuid') != data['id'] or study.get('schema_version') != 1:
            raise BookError('随书学习源与正文身份不同')
        if {a['id'] for a in study['assets']} != resource_ids:
            raise BookError('随书学习源与材料清单不同')
    return data, result


def export_rich_book(root: Path, spec: dict, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w') as archive:
        def put(name, raw, method):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = method
            info.external_attr = 0o644 << 16
            archive.writestr(info, raw)
        put('mimetype', b'application/epub+zip', zipfile.ZIP_STORED)
        for name in sorted(spec['epubSource']['files']):
            put(name, safe_relative(root, spec['epubSource']['root'] + '/' + name).read_bytes(), zipfile.ZIP_DEFLATED)
    return {**file_identity(output), 'chapters': len(spec['chapters']), 'fallback': 'semantic_epub_no_executable_content'}


def compiled_study(root: Path, spec: dict, epub_hash: str):
    study = json.loads(safe_relative(root, spec['studySource']).read_text(encoding='utf-8'))
    study['book_revision_sha256'] = epub_hash
    study['revision'] = spec['revision']
    by_id = {r['id']: r for r in spec['resources']}
    for asset in study['assets']:
        asset.update({k: by_id[asset['id']][k] for k in ('bytes', 'sha256')})
    for activity in study['activities']:
        activity['registered'] = False
    return study
