#!/usr/bin/env python3
"""Train an ordinary 3DGS on one finite scene's object-free background images.

The external trainer must be installed separately. This wrapper gives it only
background_training/ and attaches its final point cloud as env_gs.ply. It does
not establish geometric accuracy or alignment by itself.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__:
    from .attach_background_gs import attach, sha256_file
else:
    from attach_background_gs import attach, sha256_file


def check_background_input(scene: Path) -> tuple[Path, int]:
    """Reject far-field proxies and accidental held-out training frames."""
    scene = scene.resolve(strict=True)
    plan = json.loads((scene / 'scene_plan.json').read_text(encoding='utf-8'))
    if plan.get('background_mode') != 'finite':
        raise ValueError(
            'this wrapper requires scene_plan.json background_mode=finite; '
            'HDR-only panoramas do not provide nearby background depth')

    source = scene / 'background_training'
    train_json = source / 'transforms_train.json'
    train = json.loads(train_json.read_text(encoding='utf-8'))
    test = json.loads((source / 'transforms_test.json').read_text(encoding='utf-8'))
    if test.get('frames'):
        raise ValueError('background_training/transforms_test.json must have no frames')
    fov = float(train['camera_angle_x'])
    if (not math.isfinite(fov) or not 0 < fov < math.pi
            or not math.isclose(fov, float(plan['camera_angle_x_rad']), abs_tol=1e-6)
            or not math.isclose(fov, float(test['camera_angle_x']), abs_tol=1e-6)):
        raise ValueError('background camera FOV disagrees with scene_plan')
    expected_views = {int(view['index']): view for view in plan['views']
                      if view['role'] == 'train'}
    expected = set(expected_views)
    if len(expected) < 2:
        raise ValueError('need at least two background training camera views')
    frames = train.get('frames', [])
    paths = [str(frame['file_path']) for frame in frames]
    expected_paths = {f'./images/{index:03d}' for index in expected}
    if len(paths) != len(expected) or set(paths) != expected_paths:
        raise ValueError('background frames must match only scene_plan train views')
    for frame in frames:
        index = int(frame['file_path'].split('/')[-1])
        camera = np.asarray(frame.get('transform_matrix'), dtype=np.float64)
        planned_camera = np.asarray(expected_views[index]['c2w_blender'], dtype=np.float64)
        if (camera.shape != (4, 4) or planned_camera.shape != (4, 4)
                or not np.all(np.isfinite(camera))
                or not np.allclose(camera, planned_camera, atol=1e-6)):
            raise ValueError(f'background camera differs from scene_plan for view {index}')
        image = source / f"{frame['file_path']}.png"
        if not image.is_file():
            raise FileNotFoundError(f'missing object-free background image: {image}')
        if not image.resolve(strict=True).is_relative_to((source / 'images').resolve()):
            raise ValueError(f'background image points outside the training directory: {image}')
    if not (source / 'points3d.ply').is_file():
        raise FileNotFoundError('background_training/points3d.ply is missing')
    return source, len(frames)


def trainer_command(
    trainer_dir: Path, python: str, source: Path, model_dir: Path, iterations: int
) -> list[str]:
    if iterations <= 0:
        raise ValueError('--iterations must be positive')
    script = trainer_dir.resolve(strict=True) / 'train.py'
    if not script.is_file():
        raise FileNotFoundError(f'original 3DGS train.py not found: {script}')
    if not (Path(python).is_file() or shutil.which(python)):
        raise FileNotFoundError(f'Python executable not found: {python}')
    return [python, str(script), '-s', str(source), '-m', str(model_dir),
            '--iterations', str(iterations)]


def train_and_attach(
    scene: Path, trainer_dir: Path, model_dir: Path, python: str,
    iterations: int, copy: bool = False, dry_run: bool = False,
) -> Path | None:
    scene = scene.resolve(strict=True)
    source, view_count = check_background_input(scene)
    model_dir = model_dir.resolve()
    if model_dir == source or model_dir.is_relative_to(source):
        raise ValueError('the 3DGS model directory must be outside background_training/')
    if model_dir == scene / 'eval_only' or model_dir.is_relative_to(scene / 'eval_only'):
        raise ValueError('the 3DGS model directory must be outside eval_only/')
    if model_dir.exists() and any(model_dir.iterdir()):
        raise FileExistsError(f'model directory already contains files: {model_dir}')
    if (scene / 'env_gs.ply').exists() or (scene / 'env_gs.ply').is_symlink():
        raise FileExistsError(f'env_gs.ply already exists in {scene}')
    if (scene / 'background_gs_manifest.json').exists():
        raise FileExistsError(f'background_gs_manifest.json already exists in {scene}')
    command = trainer_command(trainer_dir, python, source, model_dir, iterations)
    print('Training background 3DGS from:', source, flush=True)
    print('Trainer command:', ' '.join(command), flush=True)
    if dry_run:
        return None

    subprocess.run(command, cwd=trainer_dir.resolve(strict=True), check=True)
    ply = model_dir / 'point_cloud' / f'iteration_{iterations}' / 'point_cloud.ply'
    if not ply.is_file():
        raise FileNotFoundError(f'3DGS trainer returned without its final point cloud: {ply}')
    provenance = (
        f'trained_on_finite_scene_background_only scene_id={scene.name} '
        f'views={view_count} iterations={iterations} '
        f'transforms_sha256={sha256_file(source / "transforms_train.json")} '
        f'trainer={trainer_dir.resolve(strict=True)}'
    )
    destination = attach(scene, ply, provenance=provenance, copy=copy)
    print('Attached background GS:', destination, flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
