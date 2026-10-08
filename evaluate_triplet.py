"""Evaluate every held-out image against the test gallery, excluding itself."""

import argparse
import hashlib
import json
from pathlib import Path

from dataset_utils import validate_manifest
from retrieval_metrics import retrieval_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='dataset_split.json')
    parser.add_argument('--image-folder', default='static/dataset')
    parser.add_argument('--model', default='triplet_base_final.h5')
    parser.add_argument('--output', default='evaluation_report.json')
    parser.add_argument('--k', type=int, nargs='+', default=[1, 5, 10])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--preparation-report', help='Optional dataset provenance and contributor metadata')
    args = parser.parse_args()
    try:
        manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
        validate_manifest(manifest, args.image_folder)
        digest = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
        if digest != manifest['model_sha256']:
            raise ValueError('Model does not match the training manifest; retrain before evaluating')
        records = manifest['splits']['test']
        if not manifest['smoke_test'] and len(records) < manifest['minimum_test_images']:
            raise ValueError('Test set is below the recorded minimum')
        from extract_features_triplet import TripletFeatureExtractor
        extractor = TripletFeatureExtractor(args.model)
        embeddings = extractor.extract_paths([Path(args.image_folder) / row['filename']
                                               for row in records], args.batch_size)
        report = retrieval_metrics(embeddings, [r['label'] for r in records],
                                   ks=args.k, seed=manifest['seed'])
        report.update({'status': 'smoke_test_only' if manifest['smoke_test'] else 'held_out_evaluation',
                       'model_sha256': digest, 'seed': manifest['seed'],
                       'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                       'split_strategy': manifest.get('split_strategy', 'image_disjoint'),
                       'split_counts': manifest['counts'],
                       'relevance': 'Same explicit category label; not human-rated visual similarity',
                       'limitations': 'Image-disjoint, same-category evaluation. Review perceptual duplicates '
                                      'and keep related product/subject images in one split. '
                                      'Passing a size guardrail does not establish reliable performance.'})
        if args.preparation_report:
            provenance = json.loads(Path(args.preparation_report).read_text(encoding='utf-8'))
            metadata = {row['filename']: row for row in provenance['records']}
            contributors = {split: {metadata[row['filename']]['sender_id']
                                    for row in manifest['splits'][split]}
                            for split in ('train', 'validation', 'test')}
            report['dataset'] = {k: provenance[k] for k in ('source', 'source_commit', 'license',
                                                           'metadata_rows', 'retained_images', 'category_counts')}
            report['excluded_images'] = len(provenance['excluded'])
            report['contributor_split_audit'] = {
                'train_contributors': len(contributors['train']),
                'validation_contributors': len(contributors['validation']),
                'test_contributors': len(contributors['test']),
                'train_test_shared_contributors': len(contributors['train'] & contributors['test']),
                'note': 'Contributor IDs are kept in one split when group labels are provided; '
                        'this does not establish product-identity or perceptual-duplicate separation.'}
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
