## Quick context

- Language: Python 3.11 (project was developed and tested on 3.11).
- Purpose: generate coverage scan paths from planar mesh segments (.stl) and visualize/refine them with Open3D.
- Key entry points:
  - `InteractiveGUI.py` — full-featured GUI (Open3D GUI). Run with `python InteractiveGUI.py [mesh_path]`.
  - `boundary_detection.py` — procedural example/CLI-style runner that uses `Best_Fit_CPP` to create paths.
  - `tester.py` — small visualization / dev playground (quick checks).

## Important files and responsibilities

- `PCAClass.py` — core algorithm and data model. `PCABounding` sets up mesh, PCA, and KD-tree. `Best_Fit_CPP` implements:
  - boundary detection (`boundary_edge_calculations`)
  - Bézier fitting (`fit_curve3d`, `edge_fitter`, `bezier_curve3N`)
  - path construction (`scan_information`, `line_interpolator`)
  - ray casting & coverage check (`ray_cast_prep`, `local_scanned_area`)
  - refinement (`Potential_Field`).
  Edit this file for algorithm-level changes.

- `InteractiveGUI.py` — UI glue for the user workflow (open file -> select corners/edges -> generate passes -> refine -> export). The UI relies on mutating `Best_Fit_CPP`'s state (e.g., `CPP.passes`, `CPP.colors`). Use this file when changing interaction flows or visualization.

- `boundary_detection.py` and `plain_Best_fit.py` (archive) — examples / scripts showing how to call `Best_Fit_CPP` from a script. Good examples for non-GUI automation.

- `plane_segments/` — place test `.stl` files here; many scripts use hard-coded/relative paths into this folder.

## Dependency and setup notes (short)

- Use the provided `requirements.txt` and Python 3.11. Example:

```powershell
pip install -r requirements.txt
```

- Key versions (explicit in `requirements.txt`): `open3d==0.18.0`, `numpy==1.26.4`, `scipy==1.14.1`, `robodk==5.7.5`.
- Warning: open3d 0.18.0 is sensitive to `numpy` major versions: avoid numpy 2.x with this Open3D build (see README.md).

## How to run common workflows

- Quick GUI (interactive):

```powershell
python InteractiveGUI.py
# or pass a mesh path: python InteractiveGUI.py plane_segments\Airfoil_surface.stl
```

- Run a scripted path generation / visualization example:

```powershell
python boundary_detection.py
```

- Quick visualization sandbox:

```powershell
python tester.py
```

## Project-specific patterns and conventions for code edits

- Stateful object model: `Best_Fit_CPP` is intentionally stateful — many functions mutate members (`edge1_CP`, `passes`, `colors`, etc.). When changing behavior, prefer to update and return state from `PCAClass.py` rather than duplicating state across files.

- Color conventions used across codebase (use these exact RGB arrays when adding visual/debug features):
  - Red [1, 0, 0] — unscanned/default mesh vertices.
  - Green [0, 1, 0] — newly-scanned points.
  - Blue [0, 0, 1] — re-scanned / overlap points.

- Bezier defaults: Bézier order is 6 and sample numbers commonly use 10 (see `Best_Fit_CPP.Bezier_order` and `Sample_num`). If you change numerical defaults, update both `PCAClass.py` and UI defaults in `InteractiveGUI.py`.

- Hard-coded file paths: Several scripts use `plane_segments\...` relative paths. When adding tests or utilities, prefer using `os.path.join(os.getcwd(), 'plane_segments', '...')` and avoid relying on implicit cwd changes.

- GUI uses Open3D's event loop and posts background tasks via `gui.Application.instance.post_to_main_thread`. Keep long-running computations in separate threads and only mutate UI state via the main thread callbacks.

## Integration points / external dependencies to be aware of

- RoboDK: `boundary_detection.py` and `tester.py` include RoboDK integration (`robodk`, `robolink`). Running this portion requires RoboDK installed and accessible from the machine; the code clears & writes targets into the RoboDK station.

- Open3D Tensors & Raycasting: `Best_Fit_CPP.ray_cast_prep()` converts legacy mesh to `o3d.t.geometry.TriangleMesh` and uses `o3d.t.geometry.RaycastingScene()`. Keep type correctness when adding code that interacts with the raycast scene.

## Examples to reference when generating changes

- To add a new refinement variant, add a method on `Best_Fit_CPP` (e.g., `refine_paths_custom`) and call it from `InteractiveGUI._on_start_refinement` — follow the `Potential_Field` pattern (returns modified `passes`).

- To change how passes are exported, update the export stub in `InteractiveGUI._on_export_paths` and mirror expected transforms shown in `tester.py` / `boundary_detection.py` where the RoboDK `Mat` is constructed.

## Small heuristics / gotchas discovered in code

- The code sometimes sets `self.tertiary_axis = np.array([0,0,1])` in ray-tracing functions (overwrites PCA-derived axis). Check `local_scanned_area` if results look unexpected.
- Many scripts mutate `cwd` temporarily when opening file dialogs (see `InteractiveGUI._on_menu_open`) — tests that rely on relative paths may fail if launched from other working directories.

---

If anything in this summary is unclear or you'd like more examples (unit test scaffolding for `PCAClass`, or a short demo script that runs the full pipeline headlessly), tell me which part to expand and I will iterate. 
