"""Verify pinned GGUF artifacts; download or convert only missing files."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parent.parent


def verify(path, manifest):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if path.stat().st_size != manifest['size_bytes'] or digest.hexdigest() != manifest['sha256']:
        raise RuntimeError(f'Model size or SHA-256 mismatch: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--converter', type=pathlib.Path, default=ROOT / 'build/wsl-cpu/bin/mini-llm')
    args = parser.parse_args()
    model_manifest = json.loads((ROOT / 'models/manifest.json').read_text())
    model = ROOT / 'models' / model_manifest['file']
    if not model.exists():
        partial = model.with_suffix('.gguf.part')
        urllib.request.urlretrieve(model_manifest['url'], partial)
        verify(partial, model_manifest)
        partial.rename(model)
    verify(model, model_manifest)
    print(f'Verified: {model}')
    if args.reference:
        manifest = json.loads((ROOT / 'models/reference-manifest.json').read_text())
        if manifest['source_sha256'] != model_manifest['sha256']:
            raise RuntimeError('Reference manifest source does not match model manifest')
        reference = ROOT / 'models' / manifest['file']
        if not reference.exists():
            partial = reference.with_suffix('.gguf.part')
            subprocess.run([str(args.converter.resolve()), '--model', str(model),
                            '--dequantize-ref', str(partial)], check=True)
            verify(partial, manifest)
            partial.rename(reference)
        verify(reference, manifest)
        print(f'Verified: {reference}')


if __name__ == '__main__':
    main()
