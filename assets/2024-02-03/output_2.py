from diffusers import DiffusionPipeline
import torch
import torch_xla.core.xla_model as xm

print(xm.xla_device())

stable_diffusion = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

stable_diffusion = stable_diffusion.to(xm.xla_device())

output = stable_diffusion(
    prompt="Gandalf with a shotgun",
    num_inference_steps=50,
)
output.images[0].save("image.png")


"""
python test.py 
/home/raix/w/diffusers/src/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
xla:0
Loading pipeline components...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00,  7.45it/s]
  0%|                                                                                                                                    | 0/50 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "test.py", line 16, in <module>
    output = stable_diffusion(
  File "/home/raix/miniconda3/envs/tpu/lib/python3.8/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/home/raix/w/diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py", line 1241, in __call__
    noise_pred = self.unet(
  File "/home/raix/miniconda3/envs/tpu/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/raix/miniconda3/envs/tpu/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/raix/w/diffusers/src/diffusers/models/unets/unet_2d_condition.py", line 1025, in forward
    add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
RuntimeError: torch_xla/csrc/convert_ops.cpp:88 : Unsupported XLA type 10
"""

Whats wrong? I quicky found some github issues that states the TPU don't support [pf16](https://github.com/pytorch/xla/issues/2917), so lets try to remove it: 

Then i quickly get this error

"""
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    output = stable_diffusion(
  File "/home/raix/miniconda3/envs/tpu/lib/python3.8/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/home/raix/w/diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py", line 1310, in __call__
    image = self.image_processor.postprocess(image, output_type=output_type)
  File "/home/raix/w/diffusers/src/diffusers/image_processor.py", line 602, in postprocess
    image = self.pt_to_numpy(image)
  File "/home/raix/w/diffusers/src/diffusers/image_processor.py", line 127, in pt_to_numpy
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
RuntimeError: Bad StatusOr access: RESOURCE_EXHAUSTED: Error loading program: Attempting to allocate 46.33M. That was not possible. There are 13.94M free.; (0x0x0_HBM0)
Error in atexit._run_exitfuncs:
Traceback (most recent call last):
  File "/home/raix/miniconda3/envs/tpu/lib/python3.8/site-packages/torch_xla/__init__.py", line 151, in _prepare_to_exit
    _XLAC._prepare_to_exit()
RuntimeError: Bad StatusOr access: RESOURCE_EXHAUSTED: Error loading program: Attempting to allocate 46.33M. That was not possible. There are 13.94M free.; (0x0x0_HBM0)
"""
Wich indicates that we try to load too much stuff into our tpu device

So Lets try with dedictated floating point precission for TPU, Brain16. With this 
```python

stable_diffusion = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16,
    variant="bf16",
)

```

now we have error that the weight's aren't saved wit this precission type. 


(tpu) raix@t1v-n-45315079-w-0:~/w$ python test.py 
/home/raix/w/diffusers/src/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
xla:0
Traceback (most recent call last):
  File "test.py", line 7, in <module>
    stable_diffusion = DiffusionPipeline.from_pretrained(
  File "/home/raix/miniconda3/envs/tpu/lib/python3.8/site-packages/huggingface_hub/utils/_validators.py", line 118, in _inner_fn
    return fn(*args, **kwargs)
  File "/home/raix/w/diffusers/src/diffusers/pipelines/pipeline_utils.py", line 1111, in from_pretrained
    cached_folder = cls.download(
  File "/home/raix/miniconda3/envs/tpu/lib/python3.8/site-packages/huggingface_hub/utils/_validators.py", line 118, in _inner_fn
    return fn(*args, **kwargs)
  File "/home/raix/w/diffusers/src/diffusers/pipelines/pipeline_utils.py", line 1726, in download
    deprecate("no variant default", "0.24.0", deprecation_message, standard_warn=False)
  File "/home/raix/w/diffusers/src/diffusers/utils/deprecation_utils.py", line 18, in deprecate
    raise ValueError(
ValueError: The deprecation tuple ('no variant default', '0.24.0', "You are trying to load the model files of the `variant=bf16`, but no such modeling files are available.The default model files: {'vae_decoder/model.onnx', 'vae_1_0/diffusion_pytorch_model.safetensors', 'text_encoder/flax_model.msgpack', 'unet/model.onnx', 'unet/diffusion_flax_model.msgpack', 'text_encoder_2/flax_model.msgpack', 'vae/diffusion_pytorch_model.safetensors', 'unet/diffusion_pytorch_model.safetensors', 'vae/diffusion_flax_model.msgpack', 'text_encoder_2/model.onnx', 'vae_encoder/model.onnx', 'text_encoder/model.safetensors', 'text_encoder_2/model.safetensors', 'text_encoder/model.onnx'} will be loaded instead. Make sure to not load from `variant=bf16`if such variant modeling files are not available. Doing so will lead to an error in v0.24.0 as defaulting to non-variantmodeling files is deprecated.") should be removed since diffusers' version 0.26.0.dev0 is >= 0.24.0


So there should be another way to do it. Googling introduces us with https://huggingface.co/blog/sdxl_jax We see that Hugging face team was using the 