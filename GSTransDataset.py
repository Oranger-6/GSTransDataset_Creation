"""Render per-scene TransparentGS inputs from known meshes.

Every view is rendered with Mitsuba and its actual sampled camera is recorded
at the same time. The finite-background mode places textured opaque surfaces
and nearby objects behind the transparent shape. Ground truth remains in
eval_only/; it is never used to fit the background 3DGS.
"""

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mitsuba as mi
import numpy as np

from projects.RefractiveCorresDataset import RefractiveCorresDataset
from scene_builder.build_utils import build_scene
from scene_builder.mitsuba_utils import set_camera_pose
from scripts.TransparentGS.prepare_nrc_scene import GL_TO_CV, write_random_ply
from utils.camera_utils import fov_to_intrinsic_mat, get_extrinsic_matrix, gen_rays
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
        cfg = convert_to_dict(conf['TransparentGS'])
        self.background_mode = cfg.get('background_mode', 'hdr')
        if self.background_mode not in ('hdr', 'finite'):
            raise ValueError('TransparentGS.background_mode must be hdr or finite')
        if self.background_mode == 'finite':
            self._init_finite(conf, cfg)
        else:
            super().__init__(conf)
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

    def _init_finite(self, conf, cfg):
        """Construct the object and opaque background without requiring HDR files."""
        self.raw_output_folder = None
        self.conf = conf
        self.preview_mode = bool(conf.get('preview', False))
        self.resource_folder = conf['Scene']['source_path']
        shape_dir = Path(self.resource_folder) / 'shape'
        self.train_shape_list = (shape_dir / 'train_shape.txt').read_text(encoding='utf-8').splitlines()
        self.test_shape_list = (shape_dir / 'test_shape.txt').read_text(encoding='utf-8').splitlines()
        self.train_shape_list = [name.strip() for name in self.train_shape_list if name.strip()]
        self.test_shape_list = [name.strip() for name in self.test_shape_list if name.strip()]
        first_shape = next(iter(self.train_shape_list or self.test_shape_list), None)
        if first_shape is None or not (shape_dir / first_shape).is_file():
            raise FileNotFoundError(f'no usable first shape listed under {shape_dir}')

        scene_conf = convert_to_dict(conf['Scene'])
        for element in scene_conf['element']:
            if element['type'] == 'transparent_mesh':
                element['mesh_filename'] = first_shape
            elif element['type'] == 'envmap_light':
                raise ValueError('finite mode uses constant ambient illumination, not an HDR file')
        if sum(e['type'] == 'transparent_mesh' for e in scene_conf['element']) != 1:
            raise ValueError('finite mode requires exactly one transparent_mesh')
        base_integrator = convert_to_dict(scene_conf.get('integrator', {'type': 'path', 'max_depth': 20}))
        scene_conf['integrator'] = {
            'type': 'aov', 'aovs': 'nn:sh_normal', 'my_image': base_integrator,
        }
