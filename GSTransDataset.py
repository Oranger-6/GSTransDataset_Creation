"""Render per-scene TransparentGS inputs from known meshes and HDR maps.

Every view is rendered with Mitsuba and its actual sampled camera is recorded
at the same time. Ground truth geometry and correspondence stay in eval_only/.
"""

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mitsuba as mi
import numpy as np

from projects.RefractiveCorresDataset import RefractiveCorresDataset
from scene_builder.mitsuba_utils import set_camera_pose
from scripts.TransparentGS.prepare_nrc_scene import GL_TO_CV, write_random_ply
from utils.camera_utils import fov_to_intrinsic_mat, get_extrinsic_matrix
from utils.registry import PROJECT_REGISTRY
from utils.tool_utils import convert_to_dict, spherical2cartesian


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def _link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        raise FileExistsError(f'will not overwrite {destination}')
    destination.symlink_to(source.resolve())


@PROJECT_REGISTRY.register()
class GSTransDataset(RefractiveCorresDataset):
    """One scene per shape/env slot with separate train and held-out views."""

    def __init__(self, conf):
        super().__init__(conf)
        cfg = convert_to_dict(conf['TransparentGS'])
        self.gs_config = cfg
        self.gs_seed = int(cfg.get('seed', 0))
        if float(conf['Scene'].get('scale', 1.0)) != 1.0:
            raise ValueError('GSTransDataset requires Scene.scale=1 to keep the GT mesh in world coordinates')
        self.heldout_views = int(cfg['heldout_views'])
        if self.camnum_per_env < 3 or not 0 < self.heldout_views < self.camnum_per_env - 1:
            raise ValueError('need at least two train views and one held-out view per scene')
        for name in ('object_init_bounds', 'background_init_bounds'):
            bounds = cfg[name]
            if len(bounds) != 6 or any(bounds[i] >= bounds[i + 3] for i in range(3)):
                raise ValueError(f'{name} must be xmin ymin zmin xmax ymax zmax')
        pose_cfg = convert_to_dict(conf['CamPose'])
        if pose_cfg.get('type') != 'RandomCamPose':
            raise ValueError('GSTransDataset currently supports RandomCamPose only')
        self.pose_cfg = pose_cfg
        self.data_root = Path(self.raw_output_folder).resolve()
        self.resource_root = Path(self.resource_folder).resolve()

    def _rng(self, split, shape_rel, env_slot):
        key = f'{self.gs_seed}|{split}|{shape_rel}|{env_slot}'.encode('utf-8')
        return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))

    def _sample_pose(self, rng):
        theta_min, theta_max = self.pose_cfg['theta_range']
        radius = rng.uniform(*self.pose_cfg['r_range'])
        cos_min = np.cos(np.deg2rad(theta_max))
        cos_max = np.cos(np.deg2rad(theta_min))
        theta = np.rad2deg(np.arccos(rng.uniform(cos_min, cos_max)))
        phi = rng.uniform(0.0, 360.0)
        position = np.asarray(spherical2cartesian(theta, phi, radius))
        return {
            'origin': (position + np.asarray(self.pose_cfg.get('shift', [0, 0, 0]))).tolist(),
            'target': list(self.pose_cfg.get('target', [0, 0, 0])),
            'up': [0, 0, 1],
        }

    def _camera_fov_and_size(self):
        params = self._cam_params
        x_fov = params['x_fov']
        try:
            x_fov = float(x_fov[0])
        except (TypeError, IndexError):
            x_fov = float(x_fov)
        width, height = (int(x) for x in params['film.size'])
        return x_fov, width, height

    def _plan(self, split, shape_rel, env_rel, slot, scene_id, rng):
        x_fov, width, height = self._camera_fov_and_size()
        K = fov_to_intrinsic_mat(x_fov, 'x', width, height)
        views = []
        for idx in range(self.camnum_per_env):
            pose = self._sample_pose(rng)
            w2c = np.eye(4)
            w2c[:3, :] = get_extrinsic_matrix(**pose)
