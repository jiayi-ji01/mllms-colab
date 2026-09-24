"""Build the compact, portable evidence pack for the cross-language report.

The module has one interface, :func:`build_evidence_pack`. It validates all
saved experiment inputs before replacing any compact result files. Model
inference and activation patching are deliberately outside this module.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np

from mllms.reporting.statistics import assert_same_ids, paired_intervals

STEPS = (
    ("step_005000", 5000),
    ("step_015000", 15000),
    ("step_030000", 30000),
    ("step_050000", 50000),
    ("step_065000", 65000),
    ("best", 77500),
)
DIRECTIONS = (
    "original_to_original",
    "clone_to_clone",
    "original_to_clone",
    "clone_to_original",
)
CONTROLS = (
    "clean",
    "opposite-number",
    "same-number-shuffled",
    "opposite-number-shuffled",
)
NULLS = ("opposite-number", "opposite-number-shuffled")
TASKS = ("simple", "pp_attractor", "object_relative", "subject_relative")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_hash(path: Path, expected: str, label: str) -> None:
    actual = digest(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: expected {expected}, found {actual}"
        )


def _assert_provenance(
    expected: dict[str, Any],
    actual: dict[str, Any],
    fields: tuple[str, ...],
    source: Path,
) -> None:
    for field in fields:
        if actual.get(field) != expected.get(field):
            raise ValueError(f"{field} differs at {source}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prompt_key(row: dict[str, Any]) -> str:
    # Group both number orientations and different verb answers to one prompt.
    return json.dumps(sorted([row['clean_prompt'], row['corrupted_prompt']]))


def summarize_columns(values, columns, records, iterations, groups):
    tasks = np.array([r['task'] for r in records])
    clusters = np.array([prompt_key(r) for r in records])
    rows = []
    for group in groups:
        mask = (
            np.ones(len(tasks), dtype=bool)
            if group == "overall"
            else tasks != "simple"
            if group == "pooled_distractor"
            else tasks == group
        )
        ordinary = paired_intervals(
            values[mask], tasks[mask], iterations=iterations
        )
        clustered = paired_intervals(
            values[mask], tasks[mask], clusters[mask], iterations=iterations
        )
        for i, column in enumerate(columns):
            rows.append(
                {
                    **column,
                    "task": group,
                    "num_examples": int(mask.sum()),
                    "num_prompt_clusters": clustered["num_clusters"],
                    "mean": float(ordinary["mean"][i]),
                    "row_ci_low": float(ordinary["ci_low"][i]),
                    "row_ci_high": float(ordinary["ci_high"][i]),
                    "cluster_ci_low": float(clustered["ci_low"][i]),
                    "cluster_ci_high": float(clustered["ci_high"][i]),
                }
            )
    return rows


def _portable_value(value: Any) -> Any:
    """Remove machine-specific prefixes from resolved experiment metadata."""
    if isinstance(value, dict):
        return {key: _portable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_value(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        normalized = value.replace("\\", "/")
        for prefix in ("artifacts/", "configs/", "data/", "outputs/"):
            marker = f"/{prefix}"
            if marker in normalized:
                return prefix + normalized.split(marker, maxsplit=1)[1]
        return Path(normalized).name
    return value


def _portable_candidates(candidates: dict[str, Any]) -> dict[str, Any]:
    result = dict(candidates)
    result.pop("selection_code_commit", None)
    result["selection_provenance"] = (
        "Frozen before test; checkpoint, dataset and config hashes are retained."
    )
    return _portable_value(result)


def build_evidence_pack(
    repo_root: Path,
    output_dir: Path,
    *,
    iterations: int = 10_000,
) -> dict[str, Any]:
    """Validate raw results and write one portable compact evidence pack."""
    root = repo_root.resolve()
    out = output_dir.resolve()
    source_files: set[Path] = set()

    def read(path):
        source_files.add(path)
        return json.loads(path.read_text())

    dataset_path = root / 'data/sva/controlled_v2/test.jsonl'
    source_files.add(dataset_path)
    dataset = read_jsonl(dataset_path)
    lookup = {r['sample_id']: r for r in dataset}
    if len(dataset) != 3200 or len(lookup) != 3200:
        raise ValueError('Expected the frozen 3200-pair test dataset')
    base = root / (
        'outputs/interpretability/'
        'wikipedia_cross_language_v2_trajectory_1280_double_null'
    )
    confirm = root / 'outputs/interpretability/wikipedia_cross_language_v2_confirm'
    reference = read(confirm / 'metadata.json')
    ids = reference['sample_ids']
    records = [lookup[i] for i in ids]
    if len(ids) != 1280 or reference['selection'] != 'all':
        raise ValueError('Expected the fixed 1280-pair all-selection cohort')
    _assert_hash(dataset_path, reference['data_sha256'], "test dataset")
    tokenizer_path = root / 'artifacts/wikipedia_tokenizer/tokenizer.model'
    source_files.add(tokenizer_path)
    _assert_hash(tokenizer_path, reference['tokenizer_sha256'], "tokenizer")
    candidate_path = root / (
        'outputs/interpretability/wikipedia_cross_language_v2_dev/'
        'candidate_manifest.json'
    )
    candidates = read(candidate_path)
    if candidates['status'] != 'frozen_before_test':
        raise ValueError('Candidate manifest was not frozen before test')
    raw_columns, raw_values, effect_columns, effect_values = [], [], [], []
    maps, metadata_rows, cohort_behavior, exploratory = [], [], [], []
    baseline_refs = {}
    tasks = np.array([r['task'] for r in records])
    for name, step in STEPS:
        directory = base / name
        meta = read(directory / 'metadata.json')
        assert_same_ids(ids, meta['sample_ids'], directory)
        if meta['checkpoint_step'] != step or meta['selection'] != 'all':
            raise ValueError(f'Unexpected checkpoint/selection at {directory}')
        _assert_provenance(
            reference,
            meta,
            ("data_sha256", "tokenizer_sha256"),
            directory,
        )
        metadata_rows.append({'checkpoint': name, 'step': step, 'checkpoint_sha256': meta['checkpoint_sha256'],
                              'num_patched_pairs': len(ids),
                              'full_test_joint_sanity_from_patching': meta['num_joint_sanity_pairs_available']})
        for direction in DIRECTIONS:
            arrays = {}
            for control in CONTROLS:
                folder = directory / direction / control
                archive_path = folder / 'per_example_scores.npz'
                source_files.add(archive_path)
                detail = read(folder / 'metadata.json')
                assert_same_ids(ids, detail['sample_ids'], folder)
                _assert_provenance(
                    meta,
                    detail,
                    ("checkpoint_sha256", "data_sha256", "tokenizer_sha256"),
                    folder,
                )
                with np.load(archive_path, allow_pickle=False) as archive:
                    assert_same_ids(ids, archive['sample_ids'], archive_path)
                    if archive['relative_positions'].tolist() != [-1]:
                        raise ValueError(f'Expected prediction position only at {folder}')
                    values = archive['head_out_delta_ld'][:, :, 0, :].astype(float)
                if values.shape != (1280, 12, 8) or not np.isfinite(values).all():
                    raise ValueError(f'Missing/nonfinite head effects at {folder}')
                arrays[control] = values
                raw_columns.append({'checkpoint': name, 'step': step, 'direction': direction, 'control': control})
                raw_values.append(values[:, 8, 3])
                if control == 'clean' and direction in DIRECTIONS[:2]:
                    examples = detail['examples']
                    assert_same_ids(ids, [x['sample_id'] for x in examples], folder)
                    clean = np.array([x['target_clean_ld'] for x in examples])
                    corrupted = np.array([x['target_corrupted_ld'] for x in examples])
                    language = direction.split('_to_')[0]
                    baseline_refs[name, language] = examples
                    for task in ['overall', 'pooled_distractor', *TASKS]:
                        mask = (np.ones(len(ids), bool) if task == 'overall' else
                                tasks != 'simple' if task == 'pooled_distractor' else tasks == task)
                        cohort_behavior.append({'checkpoint': name, 'step': step, 'language': language,
                            'task': task, 'num_pairs': int(mask.sum()),
                            'accuracy': float(((clean[mask] > 0).astype(float) + (corrupted[mask] < 0)) .mean() / 2),
                            'pair_accuracy': float(((clean[mask] > 0) & (corrupted[mask] < 0)).mean())})
            for null in NULLS:
                difference = arrays['clean'] - arrays[null]
                effect_columns.append({'checkpoint': name, 'step': step, 'direction': direction, 'control': null})
                effect_values.append(difference[:, 8, 3])
                for task in ['pooled_distractor', *TASKS]:
                    mask = tasks != 'simple' if task == 'pooled_distractor' else tasks == task
                    means = difference[mask].mean(axis=0)
                    for candidate in candidates['sites']:
                        layer, head = candidate['layer'], candidate['head']
                        exploratory.append({'checkpoint': name, 'step': step, 'direction': direction, 'control': null,
                            'task': task, 'site': candidate['site'], 'selection': candidate['selection'],
                            'num_examples': int(mask.sum()), 'mean': float(means[layer, head])})
                    if task == 'pooled_distractor' and direction in DIRECTIONS[2:] and null == NULLS[1]:
                        for layer in range(12):
                            for head in range(8):
                                maps.append({'checkpoint': name, 'step': step, 'direction': direction,
                                             'layer': layer, 'head': head, 'mean': float(means[layer, head])})
        print(f'Validated 16 archives: {name}', flush=True)

    effects = np.column_stack(effect_values)
    print('Computing paired trajectory intervals...', flush=True)
    effect_rows = summarize_columns(
        effects,
        effect_columns,
        records,
        iterations,
        ['pooled_distractor', *TASKS],
    )
    raw_rows = []
    for task in ['pooled_distractor', *TASKS]:
        mask = tasks != 'simple' if task == 'pooled_distractor' else tasks == task
        for column, values in zip(raw_columns, raw_values):
            raw_rows.append({**column, 'task': task, 'num_examples': int(mask.sum()), 'mean_delta_ld': float(values[mask].mean())})
    # Exploratory paired change from 5k, not a test of a pre-registered onset.
    changes, change_columns = [], []
    for index, column in enumerate(effect_columns):
        if column['step'] == 5000:
            continue
        baseline = next(i for i, c in enumerate(effect_columns) if c['step'] == 5000
                        and c['direction'] == column['direction'] and c['control'] == column['control'])
        changes.append(effects[:, index] - effects[:, baseline])
        change_columns.append({**column, 'reference_step': 5000})
    change_rows = summarize_columns(
        np.column_stack(changes),
        change_columns,
        records,
        iterations,
        ['pooled_distractor'],
    )

    print('Computing full-test behavior intervals...', flush=True)
    behavior_values, behavior_columns, behavior_sources, baseline_checks = [], [], [], []
    test_ids = [r['sample_id'] for r in dataset]
    for name, step in STEPS:
        candidates_paths = [root / f'outputs/evaluation/{prefix}/{name}/sva_summary.json'
                            for prefix in ['wikipedia_sva_v2', 'sva_wikipedia_v2']]
        summary_path = next((p for p in candidates_paths if p.is_file()), None)
        if summary_path is None:
            raise FileNotFoundError(f'Full-test SVA evaluation missing: {name}')
        summary = read(summary_path)
        if summary['checkpoint_step'] != step or summary['overall']['num_pairs'] != 3200 or summary['num_rejected']:
            raise ValueError(f'Incomplete/wrong full-test SVA at {summary_path}')
        rows_path = summary_path.with_name('sva_results.jsonl')
        source_files.add(rows_path)
        score_rows = read_jsonl(rows_path)
        score_map = {r['sample_id']: r for r in score_rows}
        if len(score_rows) != 3200 or set(score_map) != set(test_ids):
            raise ValueError(f'Full-test SVA IDs differ at {rows_path}')
        for r in score_rows:
            for field in ['clean_prompt', 'corrupted_prompt', 'clean_answer', 'corrupted_answer']:
                if r[field] != lookup[r['sample_id']][field]:
                    raise ValueError(f'SVA dataset text differs: {rows_path}')
        ordered = [score_map[i] for i in test_ids]
        for language in ['original', 'clone']:
            clean = np.array([r[language]['clean_ld'] for r in ordered])
            corrupted = np.array([r[language]['corrupted_ld'] for r in ordered])
            max_difference = max(abs(example[f'target_{variant}_ld'] -
                                     score_map[example['sample_id']][language][f'{variant}_ld'])
                                 for example in baseline_refs[name, language] for variant in ['clean', 'corrupted'])
            if max_difference > 1e-4:
                raise ValueError(f'SVA and patching baselines disagree: {name}/{language}')
            baseline_checks.append({'checkpoint': name, 'language': language,
                                    'max_absolute_ld_difference': max_difference, 'tolerance': 1e-4})
            metrics = {'accuracy': ((clean > 0).astype(float) + (corrupted < 0)) / 2,
                       'pair_accuracy': ((clean > 0) & (corrupted < 0)).astype(float),
                       'oriented_ld': (clean - corrupted) / 2}
            if not np.isclose(metrics['accuracy'].mean(), summary['overall'][language]['accuracy']):
                raise ValueError(f'SVA summary and rows disagree: {summary_path}')
            for metric, values in metrics.items():
                behavior_columns.append({'checkpoint': name, 'step': step, 'language': language, 'metric': metric})
                behavior_values.append(values)
        behavior_sources.append({'checkpoint': name, 'step': step, 'path': str(summary_path.relative_to(root)),
                                 'joint_sanity_pairs': summary['overall']['joint_sanity_pairs'], 'device': summary['device']})
    behavior_rows = summarize_columns(np.column_stack(behavior_values), behavior_columns, dataset, iterations,
                                     ['overall', 'pooled_distractor', *TASKS])

    # Preserve the historical confirmatory statistics, without changing their inference family.
    historical = []
    for folder in ['statistics', 'statistics_opposite_number_shuffled']:
        path = confirm / folder / 'prediction_head_statistics.csv'
        source_files.add(path)
        with path.open() as stream:
            for row in csv.DictReader(stream):
                if row['layer'] == '8' and row['head'] == '3':
                    historical.append(row)
                    match = next(r for r in effect_rows if r['checkpoint'] == 'best'
                                 and r['direction'] == row['direction'] and r['control'] == row['control']
                                 and r['task'] == row['task'])
                    if not np.isclose(match['mean'], float(row['mean_difference']), atol=2e-5, rtol=1e-4):
                        raise ValueError('Best trajectory does not reproduce confirmatory mean')
    coverage = [{'task': task, 'test_pairs': sum(r['task'] == task for r in dataset),
                 'patched_pairs': int((tasks == task).sum()),
                 'coverage': float((tasks == task).sum() / sum(r['task'] == task for r in dataset)),
                 'prompt_clusters': len({prompt_key(r) for r in records if r['task'] == task})} for task in TASKS]
    pack = {'trajectory_effects.csv': effect_rows, 'trajectory_changes_from_5k.csv': change_rows,
            'behavior_full_test.csv': behavior_rows, 'behavior_fixed_cohort.csv': cohort_behavior,
            'control_means.csv': raw_rows, 'exploratory_heads.csv': exploratory, 'head_maps.csv': maps,
            'historical_confirmatory.csv': historical, 'coverage.csv': coverage}
    training_path = root / 'outputs/runs/gpt12_wikipedia_clone/train_log.jsonl'
    source_files.add(training_path)
    training = read_jsonl(training_path)
    validation = [r for r in training if r['type'] == 'validation']
    pack['training_validation.csv'] = [{k: r[k] for k in ['step', 'tokens_seen', 'original_loss',
                                      'clone_loss', 'original_perplexity', 'clone_perplexity']} for r in validation]
    ppl_path = root / 'outputs/evaluation/test_ppl_wikipedia_v2/best/test_loss_perplexity.csv'
    source_files.add(ppl_path)
    with ppl_path.open() as stream:
        pack['test_perplexity.csv'] = list(csv.DictReader(stream))
    config_path = root / 'outputs/runs/gpt12_wikipedia_clone/resolved_config.json'
    training_config = _portable_value(read(config_path))
    # Verify the checkpoint identities used by both archived patching and SVA.
    checkpoint_files = [root / 'outputs/runs/gpt12_wikipedia_clone' /
                        ('best.pt' if name == 'best' else f'checkpoints/{name}.pt') for name, _ in STEPS]
    for path, metadata in zip(checkpoint_files, metadata_rows):
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(chunk)
        sha = hasher.hexdigest()
        if sha != metadata['checkpoint_sha256']:
            raise ValueError(f'Checkpoint changed since patching: {path}')
    manifest = {'dataset': 'mllms_controlled_sva_v2', 'data_sha256': reference['data_sha256'],
                'tokenizer_sha256': reference['tokenizer_sha256'], 'training_seed': 42,
                'bootstrap_seed': 42, 'iterations': iterations, 'checkpoints': metadata_rows,
                'full_test_behavior': behavior_sources, 'primary_site': 'L8H3 (zero-based indices)',
                'training_config': training_config,
                'training_final': [r for r in training if r['type'] == 'train'][-1],
                'baseline_consistency': baseline_checks,
                'environment': {'python': platform.python_version(), **{package: importlib.metadata.version(package)
                                for package in ['numpy', 'matplotlib', 'torch', 'nbformat', 'nbconvert']}},
                'cohort': {'test_pairs': 3200, 'patched_pairs': 1280, 'pooled_distractor_pairs': 888,
                           'coverage': .4, 'selection': 'all; fixed across checkpoints, no success filtering'},
                'inference': 'Trajectory CIs are exploratory pointwise 95% percentile intervals; no onset or multiplicity-adjusted trajectory significance claim. Prompt-cluster intervals are sensitivity checks, not seed uncertainty. Historical best p values retain the original per-site/direction/null three-task Holm family; zero means no tail draws, not exact p=0.',
                'source_hashes': {str(p.relative_to(root)): digest(p) for p in sorted(source_files)}}
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in pack.items():
        csv_write(out / filename, rows)
    portable_candidates = _portable_candidates(candidates)
    (out / 'candidate_manifest.json').write_text(
        json.dumps(portable_candidates, indent=2) + '\n'
    )
    manifest['analysis_code_hashes'] = {
        str(p.relative_to(root)): digest(p)
        for p in [
            Path(__file__).resolve(),
            root / 'src/mllms/reporting/statistics.py',
        ]
    }
    manifest['compact_hashes'] = {p.name: digest(p) for p in sorted(out.glob('*.csv'))}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Complete evidence pack: {out}', flush=True)
    return manifest
