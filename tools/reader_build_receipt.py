"""Bind the shared reader's compiled files to this checkout's current sources.

This local build receipt records provenance and file integrity. It does not
certify reading quality, browser behavior, or an installed desktop application.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / 'desktop'
RECEIPT = 'reader-build-receipt.json'
SOURCE_FILES = ('desktop/index.html', 'desktop/package.json',
                'desktop/package-lock.json', 'desktop/tsconfig.json',
                'desktop/vite.config.ts')
SOURCE_TREES = ('desktop/src', 'desktop/public', 'site')


class ReaderBuildError(ValueError):
    pass


def _file_row(path: Path, root: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
            'sha256': digest.hexdigest()}


def _tree_rows(root: Path, paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ReaderBuildError(f'共享阅读器来源或构建文件无效：{path}')
        rows.append(_file_row(path, root))
    return sorted(rows, key=lambda row: row['path'].encode('utf-8'))


def source_rows() -> list[dict]:
    paths = [ROOT / name for name in SOURCE_FILES]
    for name in SOURCE_TREES:
        tree = ROOT / name
        if not tree.is_dir() or tree.is_symlink():
            raise ReaderBuildError(f'共享阅读器源码目录不存在或为链接：{name}')
        paths.extend(path for path in tree.rglob('*') if path.is_file() or path.is_symlink())
    return _tree_rows(ROOT, paths)


def build_rows(dist: Path) -> list[dict]:
    if not dist.is_dir() or dist.is_symlink():
        raise ReaderBuildError('共享阅读器构建目录不存在或为链接')
    paths = [path for path in dist.rglob('*') if path.is_file() or path.is_symlink()]
    paths = [path for path in paths if path != dist / RECEIPT]
    rows = _tree_rows(dist, paths)
    names = {row['path'] for row in rows}
    if 'index.html' not in names or not any(name.startswith('assets/') and name.endswith('.js') for name in names):
        raise ReaderBuildError('共享阅读器构建缺少入口或编译脚本')
    index = (dist / 'index.html').read_text(encoding='utf-8')
    if '<main id="app"' not in index or '<title>舒适阅读书库</title>' not in index:
        raise ReaderBuildError('构建入口不是当前共享阅读器页面')
    return rows


def _digest(rows: list[dict]) -> str:
    data = json.dumps(rows, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def _expected(dist: Path, sources: list[dict] | None = None) -> dict:
    if sources is None:
        sources = source_rows()
    files = build_rows(dist)
    return {'schemaVersion': 1, 'product': 'comfortable-reader-shared',
            'buildCommand': 'npm run build (cwd=desktop)',
            'sourceFiles': sources, 'sourceTreeSha256': _digest(sources),
            'buildFiles': files, 'buildTreeSha256': _digest(files),
            'scope': 'local source and compiled file provenance; not UI or installed-app verification'}


def build_shared_reader() -> dict:
    npm = shutil.which('npm.cmd' if os.name == 'nt' else 'npm')
    if not npm or not (DESKTOP / 'node_modules').is_dir():
        raise ReaderBuildError('共享阅读器依赖尚未准备；先在 desktop 执行 npm ci')
    before = source_rows()
    try:
        subprocess.run([npm, 'run', 'build'], cwd=DESKTOP, check=True)
    except subprocess.CalledProcessError as error:
        raise ReaderBuildError('当前 checkout 的共享阅读器构建未通过') from error
    after = source_rows()
    if after != before:
        raise ReaderBuildError('共享阅读器源码在构建过程中改变；此构建不能获得来源收据')
    dist = DESKTOP / 'dist'
    receipt = _expected(dist, after)
    (dist / RECEIPT).write_text(json.dumps(receipt, ensure_ascii=False,
                                          sort_keys=True, separators=(',', ':')) + '\n',
                                      encoding='utf-8', newline='\n')
    return verify_reader_dist(dist)


def verify_reader_dist(dist: Path) -> dict:
    dist = dist.resolve(strict=True)
    path = dist / RECEIPT
    if not path.is_file() or path.is_symlink():
        raise ReaderBuildError('共享阅读器缺少当前构建的来源收据；请在当前 checkout 运行 reader_build_receipt.py build')
    try:
        actual = json.loads(path.read_text(encoding='utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReaderBuildError('共享阅读器构建收据无法读取') from error
    expected = _expected(dist)
    if actual != expected:
        raise ReaderBuildError('共享阅读器来源或构建文件与收据不符；请在当前 checkout 重新构建')
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('build', help='build desktop/dist from this checkout, then write its receipt')
    verify = commands.add_parser('verify', help='read-only verification of a reviewed reader dist')
    verify.add_argument('--dist', type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = build_shared_reader() if args.command == 'build' else verify_reader_dist(args.dist)
    except (ReaderBuildError, FileNotFoundError, OSError, subprocess.CalledProcessError) as error:
        print(f'共享阅读器构建停止：{error}', file=sys.stderr)
        return 2
    print(json.dumps({'status': 'source_and_files_verified',
                      'source_sha256': receipt['sourceTreeSha256'],
                      'build_sha256': receipt['buildTreeSha256'],
                      'files': len(receipt['buildFiles'])}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
