python test.py
/home/raix/w/diffusers/src/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
/home/raix/w/diffusers/src/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
/home/raix/w/diffusers/src/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
Traceback (most recent call last):
  File "/home/raix/w/diffusers/src/diffusers/utils/import_utils.py", line 704, in _get_module
    return importlib.import_module("." + module_name, self.__name__)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/raix/miniconda3/envs/tpu/lib/python3.11/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 940, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/home/raix/w/diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py", line 63, in <module>
    import torch_xla.core.xla_model as xm
  File "/home/raix/miniconda3/envs/tpu/lib/python3.11/site-packages/torch_xla/__init__.py", line 142, in <module>
    import _XLAC
ImportError: libpython3.11.so.1.0: cannot open shared object file: No such file or directory

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/raix/w/test.py", line 4, in <module>
    stable_diffusion = DiffusionPipeline.from_pretrained(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/raix/miniconda3/envs/tpu/lib/python3.11/site-packages/huggingface_hub/utils/_validators.py", line 118, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/raix/w/diffusers/src/diffusers/pipelines/pipeline_utils.py", line 1111, in from_pretrained
    cached_folder = cls.download(
                    ^^^^^^^^^^^^^
  File "/home/raix/miniconda3/envs/tpu/lib/python3.11/site-packages/huggingface_hub/utils/_validators.py", line 118, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/raix/w/diffusers/src/diffusers/pipelines/pipeline_utils.py", line 1783, in download
    pipeline_class = _get_pipeline_class(
                     ^^^^^^^^^^^^^^^^^^^^
  File "/home/raix/w/diffusers/src/diffusers/pipelines/pipeline_utils.py", line 401, in _get_pipeline_class
    pipeline_cls = getattr(diffusers_module, class_name)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/raix/w/diffusers/src/diffusers/utils/import_utils.py", line 695, in __getattr__
    value = getattr(module, name)
            ^^^^^^^^^^^^^^^^^^^^^
  File "/home/raix/w/diffusers/src/diffusers/utils/import_utils.py", line 695, in __getattr__
    value = getattr(module, name)
            ^^^^^^^^^^^^^^^^^^^^^
  File "/home/raix/w/diffusers/src/diffusers/utils/import_utils.py", line 694, in __getattr__
    module = self._get_module(self._class_to_module[name])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/raix/w/diffusers/src/diffusers/utils/import_utils.py", line 706, in _get_module
    raise RuntimeError(
RuntimeError: Failed to import diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl because of the following error (look up to see its traceback):
libpython3.11.so.1.0: cannot open shared object file: No such file or directory
