# GSTransDatasetCreation

Synthetic multi-view scenes for per-scene transparent-object reconstruction with [TransparentGS](https://github.com/LetianHuang/transparentgs). This repository extends [RCDatasetCreation](https://github.com/arkgao/RCDatasetCreation) under its MIT license. The original documentation remains in [UPSTREAM_README.md](UPSTREAM_README.md).

We render again from source BasicShape and OmniObject3D meshes and HDR lighting maps. Each newly sampled camera is saved together with its image. The exact random poses of already released NRC images cannot be recovered from their filenames. No bulk mesh or HDR collection is included.

## Generated scene

Each mesh and sampled HDR environment gets its own scene:

- Transparent RGB images, binary object masks and no-object background RGB.
- A frozen scene_plan.json with every camera intrinsic matrix K, OpenCV world-to-camera matrix, TransparentGS-compatible camera-to-world matrix, mesh/HDR paths and index of refraction (IOR).
- Distinct training and held-out views; the held-out views are not read by the public training loaders.
- Two explicit random point clouds, one for the object and one for a background 3DGS.
- Ground-truth mesh, IOR, HDR lighting, normal maps and refraction correspondences in eval_only/ for evaluation only.

The background source here is a distant HDR map. It supplies background *color*, but no measured finite scene depth. A 3DGS trained on those background pictures is an **appearance proxy**, not authenticated three-dimensional environment geometry.

## Bundled smoke example

Install [requirements.txt](requirements.txt), then render the bundled cow and HDR map on CPU:

    python -m pip install -r requirements.txt
    python render_dataset.py --conf configs/gstrans_demo.yaml --device cpu

The resulting folder is result/gstrans_demo/train/<scene_id>/. This small example has three training views and one held-out view. It checks data export, not reconstruction accuracy.

## BasicShape + OmniObject3D preparation

For raw asset preparation, install the extra tools and follow [scripts/DatasetCreation/README.md](scripts/DatasetCreation/README.md):

    python -m pip install -r requirements-prep.txt

The preparation scripts place source-relative mesh files under dataset_resources/shape/ and write four source indexes:

    dataset_resources/shape/train_shape_basic.txt
    dataset_resources/shape/test_shape_basic.txt
    dataset_resources/shape/train_shape_omiverse.txt
    dataset_resources/shape/test_shape_omiverse.txt

The environment preparation scripts write dataset_resources/env_map/train_env.txt and test_env.txt. Combine and verify the shape indexes, then render:

    python scripts/TransparentGS/merge_shape_indexes.py --resource-root ./dataset_resources --sources basic omni
    python render_dataset.py --conf configs/gstrans_dataset.yaml --device gpu

The resource directory is supplied by the user and **does not exist in this repository**. [configs/gstrans_dataset.yaml](configs/gstrans_dataset.yaml) currently requests 16 views per scene, including 4 held-out views. The shape train/test split holds out different *meshes*; the view split holds out cameras of the *same scene*.

Output for one scene:

    result/gstrans_synthetic/train/<scene_id>/
      scene_plan.json                  exact inputs, IOR, K and all camera poses
      transforms_train.json           object training views, FOV in radians
      transforms_test.json            empty to prevent accidental test leakage
      images/000.png                   transparent RGB for training
      mask/000.png                     matching 0/255 silhouette
      points3d.ply                     random initialization; not the true shape
      background_training/
        images/000.png                 no-object RGB for background training
        transforms_train.json         same training cameras
        transforms_test.json          empty
        points3d.ply                   separate random initialization
      eval_only/
        gt_mesh.ply                    original source mesh
        ground_truth.json             true IOR and source paths
        train_aux/                     correspondence, valid mask, normal
        heldout/                       held-out RGB, backgrounds, masks and poses

Each plan is written before rendering and checked on resume. The scene identifier contains an environment-slot index and hash, so repeated selection of an HDR file cannot collide.

## Prepare the background 3DGS

The TransparentGS public pipeline expects env_gs.ply at the scene root. Train a standard background 3DGS on the separate background_training/ folder with [the original 3DGS trainer](https://github.com/graphdeco-inria/gaussian-splatting), then attach the resulting PLY:

    python /path/to/gaussian-splatting/train.py -s /absolute/path/to/<scene_id>/background_training -m /absolute/path/to/background_run --iterations 30000
    python scripts/TransparentGS/attach_background_gs.py --scene /absolute/path/to/<scene_id> --ply /absolute/path/to/background_run/point_cloud/iteration_30000/point_cloud.ply --provenance trained_on_this_scene_background_images

The second command validates the 3DGS PLY, links it as env_gs.ply and records its provenance. The background model and the object cameras must share a world coordinate frame and scale. After that, the public TransparentGS pipeline can use the scene root as its source path. These synthetic scenes and the public training example should not be described as a bit-for-bit reproduction of unpublished paper settings.

Check output before and after attaching background GS:

    python scripts/TransparentGS/validate_scene.py --scene /absolute/path/to/<scene_id>
    python scripts/TransparentGS/validate_scene.py --scene /absolute/path/to/<scene_id> --require-background-gs

## Already released NRC images

[prepare_nrc_scene.py](scripts/TransparentGS/prepare_nrc_scene.py) additionally adapts an existing NRC shape/environment group when a separate per-view camera bundle and background GS are supplied. It cannot infer either from the old image filenames. Its --help gives the format and options. Keep results on the original NRC release distinct from results on newly rendered scenes.

## Limits and safeguards

- The saved poses are exact for **newly generated** images only; the previous NRC release is untouched.
- Ground-truth geometry, material and refraction labels remain inside eval_only/ and must not be used for initialization, fitting or checkpoint selection.
- The new correspondence labels follow the current RCDatasetCreation inside-image filtering; they are not identical to every historical released NRC label.
- OmniObject3D mesh preprocessing does not guarantee watertight surfaces; inspect the inputs before geometrical evaluation.
- HDR background pictures cannot supply finite three-dimensional depth. Report the background proxy choice in every method comparison.
