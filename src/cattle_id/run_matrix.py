from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import yaml


def _safe_component(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def matrix_job_id(job: dict[str, object]) -> str:
    kind = str(job.get("kind", ""))
    model = _safe_component(job.get("model", "model"))
    training_seed = int(job.get("training_seed", job.get("seed", 2026)))
    split_seed = job.get("split_seed")
    augmentation_seed = job.get("augmentation_seed")
    if kind == "train":
        task = _safe_component(job.get("protocol", "protocol"))
    elif kind == "evaluate":
        evaluation = _safe_component(job.get("evaluation", "evaluation"))
        source_protocol = _safe_component(job.get("source_protocol", ""))
        task = f"{source_protocol}_{evaluation}" if source_protocol else evaluation
    else:
        raise ValueError(f"Unknown matrix job kind: {kind}")
    if split_seed is None and augmentation_seed is None:
        return f"{kind}_{model}_{task}_seed{training_seed}"
    split_seed = training_seed if split_seed is None else int(split_seed)
    augmentation_seed = training_seed if augmentation_seed is None else int(augmentation_seed)
    return (
        f"{kind}_{model}_{task}_train{training_seed}_"
        f"split{split_seed}_aug{augmentation_seed}"
    )


def select_seed_shard(
    jobs: list[dict[str, object]],
    *,
    shard_index: int,
    shard_count: int,
) -> list[dict[str, object]]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be between zero and shard_count - 1")
    seeds = sorted({int(job.get("training_seed", job.get("seed", 2026))) for job in jobs})
    selected_seeds = {seed for index, seed in enumerate(seeds) if index % shard_count == shard_index}
    return [
        job
        for job in jobs
        if int(job.get("training_seed", job.get("seed", 2026))) in selected_seeds
    ]


def load_experiment_matrix(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("experiment matrix must be a mapping")
    return data


def expand_experiment_matrix(matrix: dict[str, Any]) -> list[dict[str, object]]:
    base_config = str(matrix["base_config"])
    explicit_jobs = matrix.get("jobs")
    if explicit_jobs is not None:
        if not isinstance(explicit_jobs, list):
            raise ValueError("experiment matrix jobs must be a list")
        jobs: list[dict[str, object]] = []
        defaults = matrix.get("job_defaults", {})
        if defaults is not None and not isinstance(defaults, dict):
            raise ValueError("experiment matrix job_defaults must be a mapping")
        for raw_job in explicit_jobs:
            if not isinstance(raw_job, dict):
                raise ValueError("every explicit matrix job must be a mapping")
            job = {str(key): value for key, value in (defaults or {}).items()}
            job.update({str(key): value for key, value in raw_job.items()})
            job.setdefault("kind", "train")
            job.setdefault("base_config", base_config)
            if "training_seed" in job:
                job["training_seed"] = int(job["training_seed"])
                job.setdefault("seed", int(job["training_seed"]))
            elif "seed" in job:
                job["seed"] = int(job["seed"])
                job["training_seed"] = int(job["seed"])
            else:
                raise ValueError("explicit matrix job requires training_seed or seed")
            for key in ("split_seed", "augmentation_seed"):
                if key in job:
                    job[key] = int(job[key])
            jobs.append(job)
        identifiers = [matrix_job_id(job) for job in jobs]
        duplicates = sorted({value for value in identifiers if identifiers.count(value) > 1})
        if duplicates:
            raise ValueError(f"experiment matrix contains duplicate jobs: {duplicates}")
        return jobs

    seeds = [int(seed) for seed in matrix.get("seeds", [])]
    models = [str(model) for model in matrix.get("models", [])]
    protocols = [str(protocol) for protocol in matrix.get("protocols", [])]
    evaluations = [str(evaluation) for evaluation in matrix.get("evaluations", [])]
    evaluation_scope = str(matrix.get("evaluation_scope", "per_model_seed"))
    extended = bool(matrix.get("target_training")) or evaluation_scope == "each_training_run"

    jobs: list[dict[str, object]] = []
    for seed in seeds:
        for model in models:
            for protocol in protocols:
                train_job: dict[str, object] = {
                    "kind": "train",
                    "base_config": base_config,
                    "seed": seed,
                    "model": model,
                    "protocol": protocol,
                }
                if extended:
                    train_job["training_scope"] = "source"
                jobs.append(train_job)
                if evaluation_scope == "each_training_run":
                    for evaluation in evaluations:
                        jobs.append(
                            {
                                "kind": "evaluate",
                                "base_config": base_config,
                                "seed": seed,
                                "model": model,
                                "source_protocol": protocol,
                                "evaluation": evaluation,
                                "training_scope": "source",
                                "metadata": str(matrix.get("external_metadata", "")),
                            }
                        )
            if evaluation_scope != "each_training_run":
                for evaluation in evaluations:
                    jobs.append(
                        {
                            "kind": "evaluate",
                            "base_config": base_config,
                            "seed": seed,
                            "model": model,
                            "evaluation": evaluation,
                        }
                    )

    target = matrix.get("target_training")
    if target:
        target_base_config = str(target["base_config"])
        target_metadata = str(target["metadata"])
        target_protocols = [str(protocol) for protocol in target.get("protocols", [])]
        target_evaluations = [str(evaluation) for evaluation in target.get("evaluations", [])]
        for seed in seeds:
            for model in models:
                for protocol in target_protocols:
                    jobs.append(
                        {
                            "kind": "train",
                            "base_config": target_base_config,
                            "seed": seed,
                            "model": model,
                            "protocol": protocol,
                            "training_scope": "target",
                            "metadata": target_metadata,
                        }
                    )
                    for evaluation in target_evaluations:
                        jobs.append(
                            {
                                "kind": "evaluate",
                                "base_config": target_base_config,
                                "seed": seed,
                                "model": model,
                                "source_protocol": protocol,
                                "evaluation": evaluation,
                                "training_scope": "target",
                                "metadata": target_metadata,
                            }
                        )
    return jobs


def summarize_experiment_matrix(
    matrix: dict[str, Any],
    jobs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if jobs is None:
        jobs = expand_experiment_matrix(matrix)

    train_jobs = [job for job in jobs if job.get("kind") == "train"]
    evaluation_jobs = [job for job in jobs if job.get("kind") == "evaluate"]
    summary = {
        "base_config": str(matrix["base_config"]),
        "seed_count": len(
            {
                int(job.get("training_seed", job.get("seed", 2026)))
                for job in jobs
            }
        ),
        "model_count": len({str(job.get("model", "")) for job in jobs}),
        "protocol_count": len(
            {
                str(job.get("protocol", job.get("source_protocol", "")))
                for job in jobs
                if job.get("protocol") or job.get("source_protocol")
            }
        ),
        "evaluation_count": len(matrix.get("evaluations", [])),
        "train_jobs": len(train_jobs),
        "evaluation_jobs": len(evaluation_jobs),
        "total_jobs": len(jobs),
        "external_metadata": str(matrix.get("external_metadata", "")),
        "masking_annotations": str(matrix.get("masking_annotations", "")),
    }
    if matrix.get("target_training"):
        summary["source_train_jobs"] = len(
            [job for job in train_jobs if job.get("training_scope") == "source"]
        )
        summary["target_train_jobs"] = len(
            [job for job in train_jobs if job.get("training_scope") == "target"]
        )
    return summary


def render_gpu_runbook(
    matrix: dict[str, Any],
    jobs: list[dict[str, object]] | None = None,
    *,
    matrix_path: str = "configs/experiment_matrix_100.yaml",
) -> str:
    summary = summarize_experiment_matrix(matrix, jobs)
    models = ", ".join(str(model) for model in matrix.get("models", []))
    protocols = ", ".join(str(protocol) for protocol in matrix.get("protocols", []))
    evaluations = ", ".join(str(evaluation) for evaluation in matrix.get("evaluations", []))
    seeds = ", ".join(str(seed) for seed in matrix.get("seeds", []))
    purpose = str(matrix.get("purpose", ""))

    if purpose in {"paper_baseline_final", "primary_confirmatory_final"}:
        scope_note = (
            "Protocol-aligned paper replication with one shared seed and shared test metadata."
            if purpose == "paper_baseline_final"
            else "Five-seed EfficientNetV2B3 confirmation with zero-shot Holstein2025 evaluation."
        )
        return "\n".join(
            [
                "# Final-Scope A100/H100 Runbook",
                "",
                scope_note,
                "The historical 240-job matrix is not part of this final acceptance contract.",
                "",
                "## Matrix Summary",
                "",
                f"Base config: {summary['base_config']}",
                f"Seeds: {seeds}",
                f"Models: {models}",
                f"Protocols: {protocols}",
                f"Evaluations: {evaluations or 'none'}",
                f"Train jobs: {summary['train_jobs']}",
                f"Evaluation jobs: {summary['evaluation_jobs']}",
                f"Total jobs: {summary['total_jobs']}",
                "Colab Pro+ notebook: notebooks/colab_train.ipynb",
                "",
                "## Completion Contract",
                "",
                "1. Reuse completed run directories only when all required artifacts validate.",
                "2. Keep raw datasets in the local Colab runtime and evidence under Google Drive.",
                "3. Generate metrics and predictions before marking a training job complete.",
                "4. Regenerate scoped CSV/TEX tables from artifacts; do not type numerical results manually.",
                "5. Package final metrics, predictions, manifests and SHA-256 hashes for local thesis integration.",
                "",
                "## Commands",
                "",
                "```powershell",
                f"python -m cattle_id.run_matrix --matrix {matrix_path}",
                "python tools/evidence_summary.py --runs artifacts/runs --scope configs/final_evidence_scope.yaml",
                "python tools/final_preflight.py -s",
                "```",
                "",
                "## Claim Boundary",
                "",
                "The final evidence supports protocol-aligned replication, bounded transform robustness "
                "and public unseen-identity transfer. It does not establish field generalisation on new "
                "captures of the original CattleSSFR animals.",
                "",
            ]
        )

    if matrix.get("target_training"):
        target = matrix["target_training"]
        return "\n".join(
            [
                "# A100/H100 Open-Set Experiment Runbook",
                "",
                "This runbook executes the no-shortcut public unseen-identity route after "
                "same-identity external acquisition was confirmed unavailable.",
                "It does not describe Holstein2025 as a same-identity external acquisition "
                "holdout for CattleSSFR.",
                "",
                "## Matrix Summary",
                "",
                f"Source config: {summary['base_config']}",
                f"Target config: {target['base_config']}",
                f"Seeds: {seeds}",
                f"Models: {models}",
                f"Source protocols: {protocols}",
                f"Evaluations: {evaluations}, {', '.join(target.get('evaluations', []))}",
                f"Source train jobs: {summary['source_train_jobs']}",
                f"Target train jobs: {summary['target_train_jobs']}",
                f"Train jobs: {summary['train_jobs']}",
                f"Evaluation jobs: {summary['evaluation_jobs']}",
                f"Total jobs: {summary['total_jobs']}",
                f"Holstein2025 metadata: {summary['external_metadata']}",
                "Colab Pro+ notebook: notebooks/colab_train.ipynb",
                "",
                "## Required Gates",
                "",
                "1. Validate the pinned Holstein2025 snapshot and SHA-256 manifest.",
                "2. Run five seed shards with deterministic run IDs and skip-completed semantics.",
                "3. Evaluate every CattleSSFR run zero-shot on unseen Holstein2025 identities.",
                "4. Train on the 82 Holstein2025 development identities and evaluate the 20 unseen identities.",
                "5. Report CMC@1, CMC@5, mAP and animal-level bootstrap intervals.",
                "",
                "## Commands",
                "",
                "```powershell",
                "python tools/holstein2025_readiness.py",
                f"python -m cattle_id.run_matrix --matrix {matrix_path} "
                "--out artifacts/matrix/experiment_matrix_open_set_jobs.json",
                "python tools/evidence_summary.py",
                "python tools/final_preflight.py -s",
                "```",
                "",
                "## Claim Boundary",
                "",
                "Holstein2025 supports independent public unseen-identity evidence. It does not "
                "retroactively create new captures of the 311 CattleSSFR animals, and no claim "
                "may describe it as same-identity field validation.",
                "",
            ]
        )

    return "\n".join(
        [
            "# A100/H100 Experiment Runbook",
            "",
            "This runbook expands the approved 95-100/100 hardening matrix for GPU execution.",
            "No 100/100 field-generalization claim is allowed until "
            "data/external_acquisition/manifest.csv contains real independent acquisitions "
            "and the external validation readiness gate passes.",
            "",
            "## Matrix Summary",
            "",
            f"Base config: {summary['base_config']}",
            f"Seeds: {seeds}",
            f"Models: {models}",
            f"Protocols: {protocols}",
            f"Evaluations: {evaluations}",
            f"Train jobs: {summary['train_jobs']}",
            f"Evaluation jobs: {summary['evaluation_jobs']}",
            f"Total jobs: {summary['total_jobs']}",
            f"External metadata: {summary['external_metadata']}",
            f"Masking annotations: {summary['masking_annotations']}",
            "Colab Pro+ notebook: notebooks/colab_train.ipynb",
            "",
            "## Required Gates",
            "",
            "1. Build and validate the private external acquisition manifest.",
            "2. Generate SHA-256 hashes for external images and annotations.",
            "3. Run the full seed/model/protocol matrix on A100/H100.",
            "4. Evaluate external_acquisition_holdout and masking shortcut variants.",
            "5. Regenerate thesis CSV/TEX evidence before changing claims.",
            "",
            "## Commands",
            "",
            "```powershell",
            "python tools/external_validation_readiness.py "
            "--config configs/external_acquisition.yaml "
            "--out thesis/gates/external-validation-readiness.md",
            f"python -m cattle_id.run_matrix --matrix {matrix_path} "
            "--out artifacts/matrix/experiment_matrix_100_jobs.json",
            "python tools/evidence_summary.py",
            "python tools/final_preflight.py -s",
            "```",
            "",
            "## Claim Boundary",
            "",
            "The Cattely public benchmark remains supportive external evidence, not a substitute "
            "for the real external_acquisition_holdout. The final thesis claim must only use rows "
            "backed by completed jobs and generated evidence tables.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expand a CattleID experiment matrix.")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--runbook-out", type=Path)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    args = parser.parse_args(argv)

    matrix = load_experiment_matrix(args.matrix)
    jobs = expand_experiment_matrix(matrix)
    if (args.shard_index is None) != (args.shard_count is None):
        parser.error("--shard-index and --shard-count must be provided together")
    selected_jobs = jobs
    if args.shard_index is not None and args.shard_count is not None:
        selected_jobs = select_seed_shard(
            jobs,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    text = json.dumps(selected_jobs, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.runbook_out:
        args.runbook_out.parent.mkdir(parents=True, exist_ok=True)
        args.runbook_out.write_text(
            render_gpu_runbook(matrix, jobs, matrix_path=str(args.matrix)),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
