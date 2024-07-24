from typing import Any, Dict, List, Optional, Tuple
import torch
from .compress_function import (
    true_uniform_quantization_compress,
    true_uniform_quantization_decompress,
    true_outlier_quantization_compress,
    true_outlier_quantization_decompress,
    true_gear_compress,
    true_gear_decompress,
    true_gear_tokenwiseQ_compress,
    true_gear_tokenwiseQ_decompress,
    true_gear_tokenwiseQ_compress_nopq,
    true_gear_tokenwiseQ_decompress_nopq,
    true_gear_outlier_tokenwiseQ_compress_nopq,
    true_gear_outlier_tokenwiseQ_decompress_nopq
)
from .compress_function import (
    true_uniform_quantization_compress_batchwise,
    true_uniform_quantization_decompress_batchwise,
    true_outlier_quantization_compress_batchwise,
    true_outlier_quantization_decompress_batchwise,
    true_gear_compress,
    true_gear_decompress_batchwise,
    true_gear_compress_batchwise,
    true_mixedprec_compress_channelwise,
    true_mixedprec_decompress_channelwise,
    true_mixedprec_gear_tokenwise_compress,
    true_mixedprec_gear_tokenwise_decompress,
    true_mixedprec_gear_outlier_tokenwise_compress,
    true_mixedprec_gear_outlier_tokenwise_decompress,
    true_mixedprec_channel_tokenwise_compress,
    true_mixedprec_channel_tokenwise_decompress,
    true_mixedprec_compress_token_channelwise,
    true_mixedprec_token_channelwise_decompress,
    true_mixedprec_tokenwise_compress,
    true_mixedprec_tokenwise_decompress,
    true_channelwise_compress,
    true_channelwise_decompress,
    true_tokenwise_compress,
    true_tokenwise_decompress,
    true_channel_tokenwise_compress,
    true_channel_tokenwise_decompress,
    true_channel_separate_mixedprec_tokenwise_compress,
    true_channel_separate_mixedprec_tokenwise_decompress,
    true_groupwise_compress,
    true_groupwise_decompress
)

compress_function = {
    "uniform": true_uniform_quantization_compress,
    "outlier": true_outlier_quantization_compress,
    "gear": true_gear_compress,
    "uniform_batch": true_uniform_quantization_compress_batchwise,
    "outlier_batch": true_outlier_quantization_compress_batchwise,
    "gear_batch": true_gear_compress_batchwise,
    "gear_tokenwiseQ": true_gear_tokenwiseQ_compress,
    "gear_tokenwiseQ_nopq": true_gear_tokenwiseQ_compress_nopq,
    "gear_outlier_tokenwiseQ_nopq":true_gear_outlier_tokenwiseQ_compress_nopq,
    "mixed_channelwiseQ": true_mixedprec_compress_channelwise,
    "mixed_gear_tokenwiseQ": true_mixedprec_gear_tokenwise_compress,
    "mixed_gear_outlier_tokenwiseQ": true_mixedprec_gear_outlier_tokenwise_compress,
    "mixed_channel_tokenwiseQ": true_mixedprec_channel_tokenwise_compress,
    "mixed_token_channelwiseQ": true_mixedprec_compress_token_channelwise,
    "mixed_tokenwiseQ": true_mixedprec_tokenwise_compress,
    "channelwiseQ": true_channelwise_compress,
    "tokenwiseQ": true_tokenwise_compress,
    "channel_tokenwiseQ": true_channel_tokenwise_compress,
    "channel_separate_mixed_tokenwiseQ": true_channel_separate_mixedprec_tokenwise_compress,
    "groupwiseQ": true_groupwise_compress
}
decompress_function = {
    "uniform": true_uniform_quantization_decompress,
    "outlier": true_outlier_quantization_decompress,
    "gear": true_gear_decompress,
    "uniform_batch": true_uniform_quantization_decompress_batchwise,
    "outlier_batch": true_outlier_quantization_decompress_batchwise,
    "gear_batch": true_gear_decompress_batchwise,
    "gear_tokenwiseQ": true_gear_tokenwiseQ_decompress,
    "gear_tokenwiseQ_nopq": true_gear_tokenwiseQ_decompress_nopq,
    "gear_outlier_tokenwiseQ_nopq":true_gear_outlier_tokenwiseQ_decompress_nopq,
    "mixed_channelwiseQ": true_mixedprec_decompress_channelwise,
    "mixed_gear_tokenwiseQ": true_mixedprec_gear_tokenwise_decompress,
    "mixed_gear_outlier_tokenwiseQ": true_mixedprec_gear_outlier_tokenwise_decompress,
    "mixed_channel_tokenwiseQ": true_mixedprec_channel_tokenwise_decompress,
    "mixed_token_channelwiseQ": true_mixedprec_token_channelwise_decompress,
    "mixed_tokenwiseQ": true_mixedprec_tokenwise_decompress,
    "channelwiseQ": true_channelwise_decompress,
    "tokenwiseQ": true_tokenwise_decompress,
    "channel_tokenwiseQ": true_channel_tokenwise_decompress,
    "channel_separate_mixed_tokenwiseQ": true_channel_separate_mixedprec_tokenwise_decompress,
    "groupwiseQ": true_groupwise_decompress
}


def detect_infnan(input_tensor, string):
    if torch.isnan(input_tensor).any():
        print(string + "has nan")
        while True:
            pass
    if torch.isinf(input_tensor).any():
        print(string + "has inf")
        while True:
            pass


class CompressUnion:
    def __init__(self, compress_kwargs: Optional[Dict[str, Any]] = None):
        self.quantize_bit = compress_kwargs["quantize_bit"]
        self.compress_mode = compress_kwargs["compress_mode"]
        self.min = None
        self.step = None
        self.min_p = None
        self.min_q = None
        self.step_p = None
        self.step_q = None
        self.left = compress_kwargs["left"]
        self.rank = compress_kwargs["rank"]
        self.loop = compress_kwargs["loop"]
        self.dtype = None
        self.shape = None
        self.shape_p = None
        self.shape_q = None
        self.quantize_part = None
        self.values = None
        self.indices = None
        self.p_base = None
        self.q_base = None
        self.counter = 0
        self.streaming_gap = compress_kwargs["streaming_gap"]
        self.buffer = None
        self.streaming = compress_kwargs["stream"]
        self.seq_length = 0
        self.input_shape = 0

    def compress_function(self, input_tensor: torch.Tensor):
        self.dtype = input_tensor.dtype
        # detect_infnan(input_tensor,"compress input tensor")
        if self.compress_mode == "uniform":
            output, shape, min, step = compress_function[self.compress_mode](
                input_tensor, self.quantize_bit
            )
            self.quantize_part = output
            self.min = min
            self.step = step
            self.shape = shape
        elif self.compress_mode == "outlier":
            output, shape, min, step, values, indices = compress_function[
                self.compress_mode
            ](input_tensor, self.quantize_bit, self.left)
            self.quantize_part = output
            self.min = min
            self.step = step
            self.shape = shape
            self.values = values
            self.indices = indices
        elif self.compress_mode == "gear":
            output, shape, min, step, values, indices, p_base, q_base = (
                compress_function[self.compress_mode](
                    input_tensor, self.quantize_bit, self.left, self.rank, self.loop
                )
            )
            self.quantize_part = output
            self.min = min
            self.step = step
            self.shape = shape
            self.values = values
            self.indices = indices
            self.p_base = p_base
            self.q_base = q_base
        elif self.compress_mode == "uniform_batch":
            output, shape, min, step = compress_function[self.compress_mode](
                input_tensor, self.quantize_bit
            )
            self.quantize_part = output
            self.min = min
            self.step = step
            self.shape = shape
        elif self.compress_mode == "outlier_batch":
            output, shape, min, step, values, indices = compress_function[
                self.compress_mode
            ](input_tensor, self.quantize_bit, self.left)
            self.quantize_part = output
            self.min = min
            self.step = step
            self.shape = shape
            self.values = values
            self.indices = indices
        elif self.compress_mode == "gear_batch":
            output, shape, min, step, values, indices, p_base, q_base = (
                compress_function[self.compress_mode](
                    input_tensor, self.quantize_bit, self.left, self.rank, self.loop
                )
            )
            self.quantize_part = output
            self.min = min
            self.step = step
            self.shape = shape
            self.values = values
            self.indices = indices
            self.p_base = p_base
            self.q_base = q_base
        elif self.compress_mode == "gear_tokenwiseQ":

            (
                quantized_input,
                shape,
                min,
                step,
                p_base,
                q_base,
                shape_p,
                shape_q,
                min_p,
                min_q,
                scale_p,
                scale_q,
            ) = compress_function[self.compress_mode](
                input_tensor, self.quantize_bit, self.rank, self.loop
            )
            self.quantize_part = quantized_input
            self.min = min
            self.step = step
            self.shape = shape
            self.p_base = p_base
            self.q_base = q_base
            self.shape_p = shape_p
            self.shape_q = shape_q
            self.min_p = min_p
            self.min_q = min_q
            self.step_p = scale_p
            self.step_q = scale_q
        elif self.compress_mode == "gear_tokenwiseQ_nopq":
            quantized_input, shape, min, step, p_base, q_base = compress_function[
                self.compress_mode
            ](input_tensor, self.quantize_bit, self.rank, self.loop)
            self.quantize_part = quantized_input
            self.min = min
            self.step = step
            self.shape = shape
            self.p_base = p_base
            self.q_base = q_base
        elif self.compress_mode == "gear_outlier_tokenwiseQ_nopq":
            quantized_input, shape, min, step, values, indices, p_base, q_base = compress_function[
                self.compress_mode
            ](input_tensor, self.quantize_bit, self.left, self.rank, self.loop)
            self.quantize_part = quantized_input
            self.min = min
            self.step = step
            self.shape = shape
            self.values = values
            self.indices = indices
            self.p_base = p_base
            self.q_base = q_base
        # print("quantized_part_min_max:",self.quantize_part.min(),self.quantize_part.max(),"p_base_min_max:",self.min_p.min(),self.p_base[0].max(),"q_base_min_max:",self.min_q.min(),self.q_base[0].max())
        # detect_infnan(quantized_input,"compress quantized_input tensor")
        # detect_infnan(self.p_base[0],"compress p_base tensor")
        # detect_infnan(self.q_base[0],"compress q_base tensor")

    def decompress_function(self):
        if self.compress_mode == "uniform":
            output = decompress_function[self.compress_mode](
                self.quantize_part,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.dtype,
            )
        elif self.compress_mode == "outlier":
            output = decompress_function[self.compress_mode](
                self.quantize_part,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.dtype,
                self.values,
                self.indices,
            )
        elif self.compress_mode == "gear":
            output = decompress_function[self.compress_mode](
                self.quantize_part,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.dtype,
                self.values,
                self.indices,
                self.p_base,
                self.q_base,
            )
        elif self.compress_mode == "uniform_batch":
            output = decompress_function[self.compress_mode](
                self.quantize_part,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.dtype,
            )
        elif self.compress_mode == "outlier_batch":
            output = decompress_function[self.compress_mode](
                self.quantize_part,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.dtype,
                self.values,
                self.indices,
            )
        elif self.compress_mode == "gear_batch":
            output = decompress_function[self.compress_mode](
                self.quantize_part,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.dtype,
                self.values,
                self.indices,
                self.p_base,
                self.q_base,
            )
        elif self.compress_mode == "gear_tokenwiseQ":
            output = decompress_function[self.compress_mode](
                self.quantize_part,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.p_base,
                self.q_base,
                self.shape_p,
                self.shape_q,
                self.min_p,
                self.min_q,
                self.step_p,
                self.step_q,
                self.dtype,
            )
        elif self.compress_mode == "gear_tokenwiseQ_nopq":
            output = decompress_function[self.compress_mode](
                self.quantize_part,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.p_base,
                self.q_base,
                self.dtype,
            )
        elif self.compress_mode == "gear_outlier_tokenwiseQ_nopq":
            output = decompress_function[self.compress_mode](
                self.quantize_part,
                self.quantize_bit,
                self.shape,
                self.min,
                self.step,
                self.values,
                self.indices,
                self.p_base,
                self.q_base,
                self.dtype,
            )
        # detect_infnan(output,"decompress")
        return output

    def compress(self, input_tensor):
        self.seq_length = input_tensor.shape[-2]
        # print("compress",self.counter)
        self.input_shape = input_tensor.shape
        if self.streaming is True:
            if self.counter % self.streaming_gap == 0:
                self.buffer = None
                self.compress_function(input_tensor)
            else:
                extract_id = self.counter % self.streaming_gap
                self.buffer = input_tensor[:, :, -extract_id:, :].clone()

        else:
            self.compress_function(input_tensor)

    def decompress(self):
        # print("decompress",self.counter)
        if self.streaming is True:
            if self.counter % self.streaming_gap == 0:
                output = self.decompress_function()
                if self.buffer is not None:
                    output = torch.cat([output, self.buffer], dim=-2)

            else:
                output = self.decompress_function()

                output = torch.cat([output, self.buffer], dim=-2)

            self.counter += 1

        else:

            output = self.decompress_function()
        # detect_infnan(output,"decompress output")
        return output

class MixedPrecisionCompressUnion:
    def __init__(self, compress_kwargs: Optional[Dict[str, Any]] = None):
        self.quantize_bit_important = compress_kwargs["quantize_bit_important"]
        self.quantize_bit_unimportant = compress_kwargs["quantize_bit_unimportant"]
        self.counter = 0
        self.streaming_gap = compress_kwargs["streaming_gap"]
        self.buffer = None
        self.streaming = compress_kwargs["stream"]
        self.seq_length = 0
        self.input_shape = 0
        self.compress_mode = compress_kwargs["compress_mode"]
        if 'rank' in compress_kwargs.keys():
            self.rank = compress_kwargs['rank']
        if 'loop' in compress_kwargs.keys():
            self.loop = compress_kwargs['loop']
        if 'left' in compress_kwargs.keys():
            self.left = compress_kwargs['left']

    def compress_function(self, input_tensor: torch.Tensor, unimportant_ids: torch.Tensor):
        if self.compress_mode == "h2o":
            unimportant_ids = unimportant_ids.unsqueeze(-1).repeat(1,1,1,128)
            input_tensor.scatter_(2, unimportant_ids, 0)
            self.cached_data = input_tensor
            return
        
        self.dtype = input_tensor.dtype
        # detect_infnan(input_tensor,"compress input tensor")
        self.unimportant_ids = unimportant_ids
        all_ids = torch.arange(input_tensor.shape[2])

        if len(unimportant_ids.shape) == 3:
            self.important_ids = torch.empty((unimportant_ids.shape[0], unimportant_ids.shape[1], input_tensor.shape[2] - unimportant_ids.shape[2]), dtype=torch.long).to(unimportant_ids.device)
            for b in range(unimportant_ids.shape[0]):
                for h in range(unimportant_ids.shape[1]):
                    mask = torch.ones(input_tensor.shape[2], dtype=torch.bool)  # Start with a mask of all True
                    mask[unimportant_ids[b,h]] = False  # Set the positions of the selected indices to False
                    self.important_ids[b,h] = all_ids[mask]  # Extract indices where the mask is True
        elif len(unimportant_ids.shape) == 2:
            self.important_ids = torch.empty((unimportant_ids.shape[0], input_tensor.shape[2] - unimportant_ids.shape[1]), dtype=torch.long).to(unimportant_ids.device)
            for i in range(unimportant_ids.shape[0]):
                mask = torch.ones(input_tensor.shape[2], dtype=torch.bool)  # Start with a mask of all True
                mask[unimportant_ids[i]] = False  # Set the positions of the selected indices to False
                self.important_ids[i] = all_ids[mask]  # Extract indices where the mask is True
        else:
            ## This only work when unimportant_ids is 1-d
            self.important_ids = torch.tensor([i for i in all_ids if i not in unimportant_ids])

        if self.compress_mode == "mixed_channelwiseQ":
            quantized_important_data, min_important_data, step_important_data, quantized_unimportant_data, min_unimportant_data, step_unimportant_data = compress_function[self.compress_mode](
                input_tensor, self.important_ids, unimportant_ids, self.quantize_bit_important, self.quantize_bit_unimportant
            )
            self.quantize_important_data = quantized_important_data
            self.min_important_data = min_important_data
            self.step_important_data = step_important_data
            self.quantize_unimportant_data = quantized_unimportant_data
            self.min_unimportant_data = min_unimportant_data
            self.step_unimportant_data = step_unimportant_data
        elif self.compress_mode == "mixed_gear_tokenwiseQ":
            quantized_important_data, min_important_data, step_important_data, p_base_important_data, q_base_important_data, \
                quantized_unimportant_data, min_unimportant_data, step_unimportant_data, p_base_unimportant_data, q_base_unimportant_data = compress_function[self.compress_mode](input_tensor, self.important_ids, unimportant_ids, self.quantize_bit_important, self.quantize_bit_unimportant, self.rank, self.loop)
            self.quantize_important_data = quantized_important_data
            self.min_important_data = min_important_data
            self.step_important_data = step_important_data
            self.p_base_important_data = p_base_important_data
            self.q_base_important_data = q_base_important_data
            self.quantize_unimportant_data = quantized_unimportant_data
            self.min_unimportant_data = min_unimportant_data
            self.step_unimportant_data = step_unimportant_data
            self.p_base_unimportant_data = p_base_unimportant_data
            self.q_base_unimportant_data = q_base_unimportant_data
        elif self.compress_mode == "mixed_gear_outlier_tokenwiseQ":
            quantized_important_data, min_important_data, step_important_data, p_base_important_data, q_base_important_data, \
                quantized_unimportant_data, min_unimportant_data, step_unimportant_data, min_outlier_values, min_outlier_indices, \
                    p_base_unimportant_data, q_base_unimportant_data = compress_function[self.compress_mode](input_tensor, self.important_ids, unimportant_ids, self.quantize_bit_important, self.quantize_bit_unimportant, self.left, self.rank, self.loop)
            self.quantize_important_data = quantized_important_data
            self.min_important_data = min_important_data
            self.step_important_data = step_important_data
            self.p_base_important_data = p_base_important_data
            self.q_base_important_data = q_base_important_data
            self.quantize_unimportant_data = quantized_unimportant_data
            self.min_unimportant_data = min_unimportant_data
            self.step_unimportant_data = step_unimportant_data
            self.min_outlier_values = min_outlier_values
            self.min_outlier_indices = min_outlier_indices
            self.p_base_unimportant_data = p_base_unimportant_data
            self.q_base_unimportant_data = q_base_unimportant_data
        elif self.compress_mode == "mixed_channel_tokenwiseQ":
            quantized_important_data, min_important_data, step_important_data, channel_max_important, \
                quantized_unimportant_data, min_unimportant_data, step_unimportant_data, \
                    channel_max_unimportant = compress_function[self.compress_mode](input_tensor, self.important_ids, unimportant_ids, self.quantize_bit_important, self.quantize_bit_unimportant)
            self.quantize_important_data = quantized_important_data
            self.min_important_data = min_important_data
            self.step_important_data = step_important_data
            self.channel_max_important = channel_max_important
            self.quantize_unimportant_data = quantized_unimportant_data
            self.min_unimportant_data = min_unimportant_data
            self.step_unimportant_data = step_unimportant_data
            self.channel_max_unimportant = channel_max_unimportant
        elif self.compress_mode == "mixed_token_channelwiseQ":
            quantized_important_data, min_important_data, step_important_data, token_scale_important, \
                quantized_unimportant_data, min_unimportant_data, step_unimportant_data, \
                    token_scale_unimportant = compress_function[self.compress_mode](input_tensor, self.important_ids, unimportant_ids, self.quantize_bit_important, self.quantize_bit_unimportant)
            self.quantize_important_data = quantized_important_data
            self.min_important_data = min_important_data
            self.step_important_data = step_important_data
            self.token_scale_important = token_scale_important
            self.quantize_unimportant_data = quantized_unimportant_data
            self.min_unimportant_data = min_unimportant_data
            self.step_unimportant_data = step_unimportant_data
            self.token_scale_unimportant = token_scale_unimportant
        elif self.compress_mode == "mixed_tokenwiseQ":
            quantized_important_data, min_important_data, step_important_data, quantized_unimportant_data, min_unimportant_data, step_unimportant_data = compress_function[self.compress_mode](
                input_tensor, self.important_ids, unimportant_ids, self.quantize_bit_important, self.quantize_bit_unimportant
            )
            self.quantize_important_data = quantized_important_data
            self.min_important_data = min_important_data
            self.step_important_data = step_important_data
            self.quantize_unimportant_data = quantized_unimportant_data
            self.min_unimportant_data = min_unimportant_data
            self.step_unimportant_data = step_unimportant_data
        elif self.compress_mode == "channelwiseQ":
            quantized_important_data, min_important_data, step_important_data = compress_function[self.compress_mode](
                input_tensor, self.quantize_bit_important
            )
            self.quantize_important_data = quantized_important_data
            self.min_important_data = min_important_data
            self.step_important_data = step_important_data
        elif self.compress_mode == "tokenwiseQ":
            quantized_important_data, min_important_data, step_important_data = compress_function[self.compress_mode](
                input_tensor, self.quantize_bit_important
            )
            self.quantize_important_data = quantized_important_data
            self.min_important_data = min_important_data
            self.step_important_data = step_important_data
        elif self.compress_mode == "groupwiseQ":
            quantized_important_data, min_important_data, step_important_data = compress_function[self.compress_mode](
                input_tensor, self.quantize_bit_important
            )
            self.quantize_important_data = quantized_important_data
            self.min_important_data = min_important_data
            self.step_important_data = step_important_data
        elif self.compress_mode == "channel_tokenwiseQ":
            quantized_important_data, min_important_data, step_important_data, channel_scale = compress_function[self.compress_mode](
                input_tensor, self.quantize_bit_important
            )
            self.quantize_important_data = quantized_important_data
            self.min_important_data = min_important_data
            self.step_important_data = step_important_data
            self.channel_scale = channel_scale
        elif self.compress_mode == "channel_separate_mixed_tokenwiseQ":
            quantized_important_data, min_important_data, step_important_data, \
                quantized_unimportant_data, min_unimportant_data, step_unimportant_data, channel_scale = compress_function[self.compress_mode](
                input_tensor, self.important_ids, unimportant_ids, self.quantize_bit_important, self.quantize_bit_unimportant
            )
            self.quantize_important_data = quantized_important_data
            self.min_important_data = min_important_data
            self.step_important_data = step_important_data
            self.quantize_unimportant_data = quantized_unimportant_data
            self.min_unimportant_data = min_unimportant_data
            self.step_unimportant_data = step_unimportant_data
            self.channel_scale = channel_scale

    def decompress_function(self):
        if self.compress_mode == "h2o":
            output = self.cached_data
        elif self.compress_mode == "mixed_channelwiseQ":
            output = decompress_function[self.compress_mode](
                self.quantize_important_data,
                self.min_important_data,
                self.step_important_data,
                self.quantize_bit_important,
                self.quantize_unimportant_data,
                self.min_unimportant_data,
                self.step_unimportant_data,
                self.quantize_bit_unimportant,
                self.important_ids,
                self.unimportant_ids,
                self.dtype,
                self.input_shape
            )
        elif self.compress_mode == "mixed_gear_tokenwiseQ":
            output = decompress_function[self.compress_mode](
                self.quantize_important_data,
                self.min_important_data,
                self.step_important_data,
                self.quantize_bit_important,
                self.p_base_important_data,
                self.q_base_important_data,
                self.quantize_unimportant_data,
                self.min_unimportant_data,
                self.step_unimportant_data,
                self.quantize_bit_unimportant,
                self.p_base_unimportant_data,
                self.q_base_unimportant_data,
                self.important_ids,
                self.unimportant_ids,
                self.dtype,
                self.input_shape
            )
        elif self.compress_mode == "mixed_gear_outlier_tokenwiseQ":
            output = decompress_function[self.compress_mode](
                self.quantize_important_data,
                self.min_important_data,
                self.step_important_data,
                self.quantize_bit_important,
                self.p_base_important_data,
                self.q_base_important_data,
                self.quantize_unimportant_data,
                self.min_unimportant_data,
                self.step_unimportant_data,
                self.quantize_bit_unimportant,
                self.min_outlier_values,
                self.min_outlier_indices,
                self.p_base_unimportant_data,
                self.q_base_unimportant_data,
                self.important_ids,
                self.unimportant_ids,
                self.dtype,
                self.input_shape
            )       
        elif self.compress_mode == "mixed_channel_tokenwiseQ":
            output = decompress_function[self.compress_mode](
                self.quantize_important_data,
                self.min_important_data,
                self.step_important_data,
                self.quantize_bit_important,
                self.channel_max_important,
                self.quantize_unimportant_data,
                self.min_unimportant_data,
                self.step_unimportant_data,
                self.quantize_bit_unimportant,
                self.channel_max_unimportant,
                self.important_ids,
                self.unimportant_ids,
                self.dtype,
                self.input_shape
            )
        elif self.compress_mode == "mixed_token_channelwiseQ":
            output = decompress_function[self.compress_mode](
                self.quantize_important_data,
                self.min_important_data,
                self.step_important_data,
                self.quantize_bit_important,
                self.token_scale_important,
                self.quantize_unimportant_data,
                self.min_unimportant_data,
                self.step_unimportant_data,
                self.quantize_bit_unimportant,
                self.token_scale_unimportant,
                self.important_ids,
                self.unimportant_ids,
                self.dtype,
                self.input_shape
            )       
        if self.compress_mode == "mixed_tokenwiseQ":
            output = decompress_function[self.compress_mode](
                self.quantize_important_data,
                self.min_important_data,
                self.step_important_data,
                self.quantize_bit_important,
                self.quantize_unimportant_data,
                self.min_unimportant_data,
                self.step_unimportant_data,
                self.quantize_bit_unimportant,
                self.important_ids,
                self.unimportant_ids,
                self.dtype,
                self.input_shape
            )
        if self.compress_mode == "channelwiseQ":
            output = decompress_function[self.compress_mode](
                self.quantize_important_data,
                self.min_important_data,
                self.step_important_data,
                self.quantize_bit_important,
                self.dtype,
                self.input_shape
            )
        if self.compress_mode == "tokenwiseQ":
            output = decompress_function[self.compress_mode](
                self.quantize_important_data,
                self.min_important_data,
                self.step_important_data,
                self.quantize_bit_important,
                self.dtype,
                self.input_shape
            )
        if self.compress_mode == "groupwiseQ":
            output = decompress_function[self.compress_mode](
                self.quantize_important_data,
                self.min_important_data,
                self.step_important_data,
                self.quantize_bit_important,
                self.dtype,
                self.input_shape
            )
        if self.compress_mode == "channel_tokenwiseQ":
            output = decompress_function[self.compress_mode](
                self.quantize_important_data,
                self.min_important_data,
                self.step_important_data,
                self.quantize_bit_important,
                self.channel_scale,
                self.dtype,
                self.input_shape
            )
        if self.compress_mode == "channel_separate_mixed_tokenwiseQ":
            output = decompress_function[self.compress_mode](
                self.quantize_important_data,
                self.min_important_data,
                self.step_important_data,
                self.quantize_bit_important,
                self.quantize_unimportant_data,
                self.min_unimportant_data,
                self.step_unimportant_data,
                self.quantize_bit_unimportant,
                self.channel_scale,
                self.important_ids,
                self.unimportant_ids,
                self.dtype,
                self.input_shape
            )
        return output

    def compress(self, input_tensor, unimportant_ids):
        self.seq_length = input_tensor.shape[-2]
        # print("compress",self.counter)
        if self.streaming is True:
            if self.counter % self.streaming_gap == 0:
                self.buffer = None
                self.compress_function(input_tensor, unimportant_ids)
                self.input_shape = input_tensor.shape
            else:
                extract_id = self.counter % self.streaming_gap
                self.buffer = input_tensor[:, :, -extract_id:, :].clone()

        else:
            self.compress_function(input_tensor, unimportant_ids)

    def decompress(self):
        # print("decompress",self.counter)
        if self.streaming is True:
            if self.counter % self.streaming_gap == 0:
                output = self.decompress_function()
                if self.buffer is not None:
                    output = torch.cat([output, self.buffer], dim=-2)

            else:
                output = self.decompress_function()

                output = torch.cat([output, self.buffer], dim=-2)

            self.counter += 1

        else:

            output = self.decompress_function()
        # detect_infnan(output,"decompress output")
        return output
