import torch
import time

def transfer_8bit_to_2bit_batchwise(input: torch.Tensor):
    # shape
    assert input.dtype == torch.uint8
    assert input.shape[-1] % 4 == 0
    size = input.shape
    size = size[-1]
    size = int(size / 4)
    input[..., 0:size] = input[..., 0:size] + input[..., size:2*size] * pow(2, 2) + input[..., 2*size:3*size] * pow(2, 4) + input[..., 3*size:] * pow(2, 6)
    cache = input[..., 0:size].clone()
    # del input
    return cache

def transfer_2bit_to_8bit_batchwise(input: torch.Tensor):
    # shape
    assert input.dtype == torch.uint8
    size = input.shape
    low_end = input & 3
    mid_low_end = (input >> 2) & 3
    mid_high_end = (input >> 4) & 3
    high_end = (input >> 6) & 3
    output = torch.cat((low_end, mid_low_end, mid_high_end, high_end), dim=-1)
    return output

def transfer_8bit_to_4bit(input: torch.Tensor):
    # shape
    assert input.dtype == torch.uint8
    assert input.shape[-1] % 2 == 0
    size = input.shape
    size = size[-1]
    size = int(size / 2)
    input[0:size] = input[0:size] + input[size:] * pow(2, 4)
    cache = input[0:size].clone()
    del input
    return cache


def transfer_4bit_to_8bit(input: torch.Tensor):
    # shape
    assert input.dtype == torch.uint8
    size = input.shape
    low_end = input % pow(2, 4)
    high_end = (input - low_end) / pow(2, 4)
    output = torch.cat((low_end, high_end), dim=0)
    return output


def transfer_8bit_to_4bit_batchwise(input: torch.Tensor):
    # shape
    assert input.dtype == torch.uint8
    assert input.shape[-1] % 2 == 0
    size = input.shape
    size = size[-1]
    size = int(size / 2)
    input[..., 0:size] = input[..., 0:size] + input[..., size:] * pow(2, 4)
    cache = input[..., 0:size].clone()
    # del input
    return cache


def transfer_4bit_to_8bit_batchwise(input: torch.Tensor):
    # shape
    assert input.dtype == torch.uint8
    size = input.shape
    low_end = input % pow(2, 4)
    high_end = (input - low_end) / pow(2, 4)
    output = torch.cat((low_end, high_end), dim=-1)
    return output

def true_channel_wise_quantize(input: torch.Tensor, quantize_bit):
    shape = input.shape ## should be B,H,L,C
    # C = input.shape[-1]
    input = (
        input.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(shape[0], shape[2], shape[1] * shape[3])
    ) # bsz, seq_len, num_head*sep_dim
    C = shape[1] * shape[3]
    min = input.reshape(-1, C).min(dim=0, keepdim=True).values.unsqueeze(0).unsqueeze(0)
    max = input.reshape(-1, C).max(dim=0, keepdim=True).values.unsqueeze(0).unsqueeze(0)
    scale = (max - min) / (2**quantize_bit - 1)
    quantized_input = (input - min) / scale
    quantized_input = quantized_input.round_()
    quantized_input = quantized_input.to(torch.uint8)
    if quantize_bit == 4:
        quantized_input = transfer_8bit_to_4bit_batchwise(quantized_input)
    elif quantize_bit == 2:
        quantized_input = transfer_8bit_to_2bit_batchwise(quantized_input)
    # print("isnan:",torch.any(torch.isnan(returning_input)))
    # while(True):
    #     pass
    return quantized_input, shape, min, scale

def true_token_channel_wise_quantize(input: torch.Tensor, quantize_bit):
    shape = input.shape ## should be B,H,L,C
    # C = input.shape[-1]
    input = (
        input.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(shape[0], shape[2], shape[1] * shape[3])
    ) # bsz, seq_len, num_head*sep_dim
    C = shape[1] * shape[3]
    # token_scale = torch.sqrt(torch.abs(input).reshape(-1, shape[1] * shape[3]).max(dim=-1, keepdim=True).values.unsqueeze(0)) ## sqrt(max(abs))
    token_scale = torch.mean(torch.abs(input), dim=-1, keepdim=True) ## mean(abs())
    input = input / token_scale

    min = input.reshape(-1, C).min(dim=0, keepdim=True).values.unsqueeze(0).unsqueeze(0)
    max = input.reshape(-1, C).max(dim=0, keepdim=True).values.unsqueeze(0).unsqueeze(0)
    scale = (max - min) / (2**quantize_bit - 1)
    quantized_input = (input - min) / scale
    quantized_input = quantized_input.round_()
    quantized_input = quantized_input.to(torch.uint8)
    if quantize_bit == 4:
        quantized_input = transfer_8bit_to_4bit_batchwise(quantized_input)
    elif quantize_bit == 2:
        quantized_input = transfer_8bit_to_2bit_batchwise(quantized_input)

    return quantized_input, shape, min, scale, token_scale

def true_tokenwise_compress(
    input: torch.Tensor, quantize_bit
):
    shape = input.shape

    quantized_data, min_data, step_data = true_tokenwise_quantize(input, quantize_bit)

    return quantized_data, min_data, step_data

def true_tokenwise_decompress(
    quantized_data, min_data, step_data, quantize_bit, dtype, shape
):
    bsz, num_head, _, sep_dim = shape

    # Decompress important data
    if quantize_bit == 4:
        quantized_data = transfer_4bit_to_8bit_batchwise(quantized_data)
    elif quantize_bit == 2:
        quantized_data = transfer_2bit_to_8bit_batchwise(quantized_data)
    elif quantize_bit == 8:
        quantized_data = quantized_data

    quantized_data = quantized_data.type(dtype)
    dequantized_data = quantized_data * step_data + min_data
    dequantized_data = (
        dequantized_data.reshape(bsz, -1, num_head, sep_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
  
    return dequantized_data

def true_groupwise_compress(
    input: torch.Tensor, quantize_bit
):
    shape = input.shape

    quantized_data, min_data, step_data = true_groupwise_quantize(input, quantize_bit)

    return quantized_data, min_data, step_data

def true_groupwise_decompress(
    quantized_data, min_data, step_data, quantize_bit, dtype, shape
):
    bsz, num_head, _, sep_dim = shape

    # Decompress important data
    if quantize_bit == 4:
        quantized_data = transfer_4bit_to_8bit_batchwise(quantized_data)
    elif quantize_bit == 2:
        quantized_data = transfer_2bit_to_8bit_batchwise(quantized_data)
    elif quantize_bit == 8:
        quantized_data = quantized_data

    quantized_data = quantized_data.type(dtype)
    dequantized_data = quantized_data * step_data + min_data
    dequantized_data = (
        dequantized_data.reshape(bsz, num_head, -1, sep_dim)
    )
  
    return dequantized_data

def true_channel_tokenwise_compress(
    input: torch.Tensor, quantize_bit
):
    shape = input.shape

    quantized_data, min_data, step_data, channel_scale = true_channel_tokenwise_quantize(input, quantize_bit)

    return quantized_data, min_data, step_data, channel_scale

def true_channel_tokenwise_decompress(
    quantized_data, min_data, step_data, quantize_bit, channel_scale, dtype, shape
):
    dequantized_data = tokenwise_dequantization_w_channelscale(quantized_data, quantize_bit, min_data, step_data, channel_scale, shape, dtype)

    return dequantized_data

def true_channelwise_compress(
    input: torch.Tensor, quantize_bit
):
    shape = input.shape

    quantized_data, _, min_data, step_data = true_channel_wise_quantize(input, quantize_bit)

    return quantized_data, min_data, step_data

def true_channelwise_decompress(
    quantized_data, min_data, step_data, quantize_bit, dtype, shape
):
    bsz, num_head, _, sep_dim = shape

    # Decompress important data
    if quantize_bit == 4:
        quantized_data = transfer_4bit_to_8bit_batchwise(quantized_data)
    elif quantize_bit == 2:
        quantized_data = transfer_2bit_to_8bit_batchwise(quantized_data)
    elif quantize_bit == 8:
        quantized_data = quantized_data

    quantized_data = quantized_data.type(dtype)
    dequantized_data = quantized_data * step_data + min_data
    dequantized_data = (
        dequantized_data.reshape(bsz, -1, num_head, sep_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
  
    return dequantized_data

def true_mixedprec_compress_channelwise(
    input: torch.Tensor, important_ids, unimportant_ids, quantize_bit_important, quantize_bit_unimportant
):
    shape = input.shape
    bsz = shape[0]
    seq_len = shape[2]
    
    # Extract important and unimportant tokens
    if len(important_ids.shape) == 3:
        important_indices_expanded = important_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        unimportant_indices_expanded = unimportant_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        important_data = torch.gather(input, 2, important_indices_expanded)
        unimportant_data = torch.gather(input, 2, unimportant_indices_expanded)        
    elif len(important_ids.shape) == 2:
        important_indices_expanded = important_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        unimportant_indices_expanded = unimportant_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        important_data = torch.gather(input, 2, important_indices_expanded)
        unimportant_data = torch.gather(input, 2, unimportant_indices_expanded)
    elif len(important_ids.shape) == 1:
        # This only works for 1-d ids
        important_data = input[:, :, important_ids, :]
        unimportant_data = input[:,:,unimportant_ids,:]

    quantized_unimportant_data, _, min_unimportant_data, step_unimportant_data = true_channel_wise_quantize(unimportant_data, quantize_bit_unimportant)
    quantized_important_data, _, min_important_data, step_important_data = true_channel_wise_quantize(important_data, quantize_bit_important)

    return quantized_important_data, min_important_data, step_important_data, \
        quantized_unimportant_data, min_unimportant_data, step_unimportant_data

def true_mixedprec_decompress_channelwise(
    quantized_important_data, min_important_data, step_important_data, quantize_bit_important,\
        quantized_unimportant_data, min_unimportant_data, step_unimportant_data, quantize_bit_unimportant,\
            important_ids, unimportant_ids, dtype, shape
):
    bsz, num_head, _, sep_dim = shape

    # Decompress important data
    if quantize_bit_important == 4:
        important_data = transfer_4bit_to_8bit_batchwise(quantized_important_data)
    elif quantize_bit_important == 2:
        important_data = transfer_2bit_to_8bit_batchwise(quantized_important_data)
    elif quantize_bit_important == 8:
        important_data = quantized_important_data

    important_data = important_data.type(dtype)
    important_data = important_data * step_important_data + min_important_data
    important_data = (
        important_data.reshape(bsz, -1, num_head, sep_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    # Decompress unimportant data
    if quantize_bit_unimportant == 4:
        unimportant_data = transfer_4bit_to_8bit_batchwise(quantized_unimportant_data)
    elif quantize_bit_unimportant == 2:
        unimportant_data = transfer_2bit_to_8bit_batchwise(quantized_unimportant_data)
    elif quantize_bit_unimportant == 8:
        unimportant_data = quantized_unimportant_data
        
    unimportant_data = unimportant_data.type(dtype)
    unimportant_data = unimportant_data * step_unimportant_data + min_unimportant_data
    unimportant_data = (
        unimportant_data.reshape(bsz, -1, num_head, sep_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    # Merge them
    if len(important_ids.shape) == 3:
        decompressed_data = torch.zeros(shape, device=important_data.device, dtype=dtype)
        important_indices_expanded = important_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1]) # (1,H,L,D)
        unimportant_indices_expanded = unimportant_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        decompressed_data = decompressed_data.scatter_(2, important_indices_expanded, important_data)
        decompressed_data = decompressed_data.scatter_(2, unimportant_indices_expanded, unimportant_data)        
    elif len(important_ids.shape) == 2:
        decompressed_data = torch.zeros(shape, device=important_data.device, dtype=dtype)
        important_indices_expanded = important_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1]) # (1,H,L,D)
        unimportant_indices_expanded = unimportant_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        decompressed_data = decompressed_data.scatter_(2, important_indices_expanded, important_data)
        decompressed_data = decompressed_data.scatter_(2, unimportant_indices_expanded, unimportant_data)
    elif len(important_ids.shape) == 1:
        # This only works for 1-d ids
        decompressed_data[:, :, unimportant_ids, :] = unimportant_data
        decompressed_data[:, :, important_ids, :] = important_data

    return decompressed_data

def true_mixedprec_compress_token_channelwise(
    input: torch.Tensor, important_ids, unimportant_ids, quantize_bit_important, quantize_bit_unimportant
):
    shape = input.shape
    bsz = shape[0]
    seq_len = shape[2]
    
    # Extract important and unimportant tokens
    if len(important_ids.shape) == 3:
        important_indices_expanded = important_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        unimportant_indices_expanded = unimportant_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        important_data = torch.gather(input, 2, important_indices_expanded)
        unimportant_data = torch.gather(input, 2, unimportant_indices_expanded)        
    elif len(important_ids.shape) == 2:
        important_indices_expanded = important_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        unimportant_indices_expanded = unimportant_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        important_data = torch.gather(input, 2, important_indices_expanded)
        unimportant_data = torch.gather(input, 2, unimportant_indices_expanded)
    elif len(important_ids.shape) == 1:
        # This only works for 1-d ids
        important_data = input[:, :, important_ids, :]
        unimportant_data = input[:,:,unimportant_ids,:]

    quantized_unimportant_data, _, min_unimportant_data, step_unimportant_data, token_scale_unimportant_data = true_token_channel_wise_quantize(unimportant_data, quantize_bit_unimportant)
    quantized_important_data, _, min_important_data, step_important_data, token_scale_important_data = true_token_channel_wise_quantize(important_data, quantize_bit_important)

    return quantized_important_data, min_important_data, step_important_data, token_scale_important_data,\
        quantized_unimportant_data, min_unimportant_data, step_unimportant_data, token_scale_unimportant_data

def true_mixedprec_token_channelwise_decompress(quantized_important_data, min_important_data, step_important_data, quantize_bit_important, token_scale_important, \
        quantized_unimportant_data, min_unimportant_data, step_unimportant_data, quantize_bit_unimportant, token_scale_unimportant,\
            important_ids, unimportant_ids, dtype, shape):
    
    bsz, num_head, _, sep_dim = shape
    important_data_shape = torch.Size([bsz, num_head, quantized_important_data.shape[1], sep_dim])
    unimportant_data_shape = torch.Size([bsz, num_head, quantized_unimportant_data.shape[1], sep_dim])
    
    # Dequantize important data
    if quantize_bit_important == 4:
        important_data = transfer_4bit_to_8bit_batchwise(quantized_important_data)
    elif quantize_bit_important == 2:
        important_data = transfer_2bit_to_8bit_batchwise(quantized_important_data)
    elif quantize_bit_important == 8:
        important_data = quantized_important_data

    important_data = important_data.type(dtype)
    important_data = (important_data * step_important_data + min_important_data) * token_scale_important
    important_data = (
        important_data.reshape(bsz, -1, num_head, sep_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    # Dequantize unimportant data
    if quantize_bit_unimportant == 4:
        unimportant_data = transfer_4bit_to_8bit_batchwise(quantized_unimportant_data)
    elif quantize_bit_unimportant == 2:
        unimportant_data = transfer_2bit_to_8bit_batchwise(quantized_unimportant_data)
    elif quantize_bit_unimportant == 8:
        unimportant_data = quantized_unimportant_data
    unimportant_data = unimportant_data.type(dtype)
    unimportant_data = (unimportant_data * step_unimportant_data + min_unimportant_data) * token_scale_unimportant
    unimportant_data = (
        unimportant_data.reshape(bsz, -1, num_head, sep_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    # dequantized_unimportant_input = dequantized_unimportant_input * channel_max_unimportant

    # Merge them
    if len(important_ids.shape) == 3:
        decompressed_data = torch.zeros(shape, device=important_data.device, dtype=dtype)
        important_indices_expanded = important_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1]) # (1,H,L,D)
        unimportant_indices_expanded = unimportant_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        decompressed_data = decompressed_data.scatter_(2, important_indices_expanded, important_data)
        decompressed_data = decompressed_data.scatter_(2, unimportant_indices_expanded, unimportant_data)        
    elif len(important_ids.shape) == 2:
        decompressed_data = torch.zeros(shape, device=important_data.device, dtype=dtype)
        important_indices_expanded = important_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1]) # (1,H,L,D)
        unimportant_indices_expanded = unimportant_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        decompressed_data = decompressed_data.scatter_(2, important_indices_expanded, important_data)
        decompressed_data = decompressed_data.scatter_(2, unimportant_indices_expanded, unimportant_data)
    elif len(important_ids.shape) == 1:
        # This only works for 1-d ids
        decompressed_data[:, :, unimportant_ids, :] = unimportant_data
        decompressed_data[:, :, important_ids, :] = important_data

    return decompressed_data

def true_mixedprec_gear_tokenwise_compress(
    input: torch.Tensor, important_ids, unimportant_ids, quantize_bit_important, quantize_bit_unimportant, rank, loop
):
    shape = input.shape

    # Extract important and unimportant tokens
    if len(important_ids.shape) == 3:
        important_indices_expanded = important_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        unimportant_indices_expanded = unimportant_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        important_data = torch.gather(input, 2, important_indices_expanded)
        unimportant_data = torch.gather(input, 2, unimportant_indices_expanded)        
    elif len(important_ids.shape) == 2:
        important_indices_expanded = important_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        unimportant_indices_expanded = unimportant_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        important_data = torch.gather(input, 2, important_indices_expanded)
        unimportant_data = torch.gather(input, 2, unimportant_indices_expanded)
    elif len(important_ids.shape) == 1:
        # This only works for 1-d ids
        important_data = input[:, :, important_ids, :]
        unimportant_data = input[:,:,unimportant_ids,:]

    # Test equivalence of gather operation
    # important_ids_1d = important_ids[0]
    # important_data_tmp = input[:, :, important_ids_1d, :]
    # print(important_data.equal(important_data_tmp))

    quantized_important_data, shape_important_data, min_important_data, step_important_data, \
         p_base_important_data, q_base_important_data = true_gear_tokenwiseQ_compress_nopq(important_data, quantize_bit_important, rank, loop)
    
    quantized_unimportant_data, shape_unimportant_data, min_unimportant_data, step_unimportant_data, \
         p_base_unimportant_data, q_base_unimportant_data = true_gear_tokenwiseQ_compress_nopq(unimportant_data, quantize_bit_unimportant, rank, loop)

    return quantized_important_data, min_important_data, step_important_data, p_base_important_data, q_base_important_data, \
        quantized_unimportant_data, min_unimportant_data, step_unimportant_data, p_base_unimportant_data, q_base_unimportant_data

def true_mixedprec_gear_tokenwise_decompress(quantized_important_data, min_important_data, step_important_data, quantize_bit_important, p_base_important_data, q_base_important_data, \
        quantized_unimportant_data, min_unimportant_data, step_unimportant_data, quantize_bit_unimportant, p_base_unimportant_data, q_base_unimportant_data, \
            important_ids, unimportant_ids, dtype, shape):
    bsz, num_head, seq_len, sep_dim = shape
    important_data_shape = torch.Size([bsz, num_head, quantized_important_data.shape[1], sep_dim])
    unimportant_data_shape = torch.Size([bsz, num_head, quantized_unimportant_data.shape[1], sep_dim])
    # Dequantize important data
    dequantized_important_input = tokenwise_dequantization(
        quantized_important_data, quantize_bit_important, min_important_data, step_important_data, important_data_shape, dtype
    )
    error = q_base_important_data[0] @ torch.transpose(p_base_important_data[0], 1, 2)
    batch, num_head, seq_len, sep_dim = dequantized_important_input.shape
    error = error.reshape(batch, seq_len, num_head, sep_dim)
    error = error.permute(0, 2, 1, 3)
    dequantized_important_input = dequantized_important_input + error.to(dtype)

    # Dequantize unimportant data
    dequantized_unimportant_input = tokenwise_dequantization(
        quantized_unimportant_data, quantize_bit_unimportant, min_unimportant_data, step_unimportant_data, unimportant_data_shape, dtype
    )
    error = q_base_unimportant_data[0] @ torch.transpose(p_base_unimportant_data[0], 1, 2)
    batch, num_head, seq_len, sep_dim = dequantized_unimportant_input.shape
    error = error.reshape(batch, seq_len, num_head, sep_dim)
    error = error.permute(0, 2, 1, 3)
    dequantized_unimportant_input = dequantized_unimportant_input + error.to(dtype)

    # Merge them
    if len(important_ids.shape) == 3:
        decompressed_data = torch.zeros(shape, device=dequantized_important_input.device, dtype=dtype)
        important_indices_expanded = important_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1]) # (1,H,L,D)
        unimportant_indices_expanded = unimportant_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        decompressed_data = decompressed_data.scatter_(2, important_indices_expanded, dequantized_important_input)
        decompressed_data = decompressed_data.scatter_(2, unimportant_indices_expanded, dequantized_unimportant_input)        
    elif len(important_ids.shape) == 2:
        decompressed_data = torch.zeros(shape, device=dequantized_important_input.device, dtype=dtype)
        important_indices_expanded = important_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1]) # (1,H,L,D)
        unimportant_indices_expanded = unimportant_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        decompressed_data = decompressed_data.scatter_(2, important_indices_expanded, dequantized_important_input)
        decompressed_data = decompressed_data.scatter_(2, unimportant_indices_expanded, dequantized_unimportant_input)
    elif len(important_ids.shape) == 1:
        # This only works for 1-d ids
        decompressed_data[:, :, unimportant_ids, :] = dequantized_unimportant_input
        decompressed_data[:, :, important_ids, :] = dequantized_important_input

    return decompressed_data

def true_mixedprec_gear_outlier_tokenwise_compress(
    input: torch.Tensor, important_ids, unimportant_ids, quantize_bit_important, quantize_bit_unimportant, left, rank, loop
):
    shape = input.shape

    # Extract important and unimportant tokens
    if len(important_ids.shape) == 3:
        important_indices_expanded = important_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        unimportant_indices_expanded = unimportant_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        important_data = torch.gather(input, 2, important_indices_expanded)
        unimportant_data = torch.gather(input, 2, unimportant_indices_expanded)        
    elif len(important_ids.shape) == 2:
        important_indices_expanded = important_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        unimportant_indices_expanded = unimportant_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        important_data = torch.gather(input, 2, important_indices_expanded)
        unimportant_data = torch.gather(input, 2, unimportant_indices_expanded)
    elif len(important_ids.shape) == 1:
        # This only works for 1-d ids
        important_data = input[:, :, important_ids, :]
        unimportant_data = input[:,:,unimportant_ids,:]

    # Test equivalence of gather operation
    # important_ids_1d = important_ids[0]
    # important_data_tmp = input[:, :, important_ids_1d, :]
    # print(important_data.equal(important_data_tmp))

    quantized_important_data, shape_important_data, min_important_data, step_important_data, \
         p_base_important_data, q_base_important_data = true_gear_tokenwiseQ_compress_nopq(important_data, quantize_bit_important, rank, loop)
    
    quantized_unimportant_data, shape_unimportant_data, min_unimportant_data, step_unimportant_data, min_outlier_values, min_outlier_indices,\
         p_base_unimportant_data, q_base_unimportant_data = true_gear_outlier_tokenwiseQ_compress_nopq(unimportant_data, quantize_bit_unimportant, left, rank, loop)

    return quantized_important_data, min_important_data, step_important_data, p_base_important_data, q_base_important_data, \
        quantized_unimportant_data, min_unimportant_data, step_unimportant_data, min_outlier_values, min_outlier_indices, p_base_unimportant_data, q_base_unimportant_data


def true_mixedprec_gear_outlier_tokenwise_decompress(quantized_important_data, min_important_data, step_important_data, quantize_bit_important, p_base_important_data, q_base_important_data, \
        quantized_unimportant_data, min_unimportant_data, step_unimportant_data, quantize_bit_unimportant, min_outlier_values, min_outlier_indices, p_base_unimportant_data, q_base_unimportant_data, \
            important_ids, unimportant_ids, dtype, shape):
    bsz, num_head, _, sep_dim = shape
    important_data_shape = torch.Size([bsz, num_head, quantized_important_data.shape[1], sep_dim])
    unimportant_data_shape = torch.Size([bsz, num_head, quantized_unimportant_data.shape[1], sep_dim])
    # Dequantize important data
    dequantized_important_input = tokenwise_dequantization(
        quantized_important_data, quantize_bit_important, min_important_data, step_important_data, important_data_shape, dtype
    )
    error = q_base_important_data[0] @ torch.transpose(p_base_important_data[0], 1, 2)
    batch, num_head, seq_len, sep_dim = dequantized_important_input.shape
    error = error.reshape(batch, seq_len, num_head, sep_dim)
    error = error.permute(0, 2, 1, 3)
    dequantized_important_input = dequantized_important_input + error.to(dtype)

    # Dequantize unimportant data
    dequantized_unimportant_input = tokenwise_dequantization(
        quantized_unimportant_data, quantize_bit_unimportant, min_unimportant_data, step_unimportant_data, unimportant_data_shape, dtype
    )
    # Merge outliers
    unimportant_input_shape = dequantized_unimportant_input.shape
    dequantized_unimportant_input = dequantized_unimportant_input.reshape(bsz, -1)
    dequantized_unimportant_input.scatter_(1, min_outlier_indices, min_outlier_values)
    dequantized_unimportant_input = dequantized_unimportant_input.reshape(unimportant_input_shape)

    error = q_base_unimportant_data[0] @ torch.transpose(p_base_unimportant_data[0], 1, 2)
    batch, num_head, seq_len, sep_dim = dequantized_unimportant_input.shape
    error = error.reshape(batch, seq_len, num_head, sep_dim)
    error = error.permute(0, 2, 1, 3)
    dequantized_unimportant_input = dequantized_unimportant_input + error.to(dtype)

    # Merge them
    if len(important_ids.shape) == 3:
        decompressed_data = torch.zeros(shape, device=dequantized_important_input.device, dtype=dtype)
        important_indices_expanded = important_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1]) # (1,H,L,D)
        unimportant_indices_expanded = unimportant_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        decompressed_data = decompressed_data.scatter_(2, important_indices_expanded, dequantized_important_input)
        decompressed_data = decompressed_data.scatter_(2, unimportant_indices_expanded, dequantized_unimportant_input)        
    elif len(important_ids.shape) == 2:
        decompressed_data = torch.zeros(shape, device=dequantized_important_input.device, dtype=dtype)
        important_indices_expanded = important_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1]) # (1,H,L,D)
        unimportant_indices_expanded = unimportant_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        decompressed_data = decompressed_data.scatter_(2, important_indices_expanded, dequantized_important_input)
        decompressed_data = decompressed_data.scatter_(2, unimportant_indices_expanded, dequantized_unimportant_input)
    elif len(important_ids.shape) == 1:
        # This only works for 1-d ids
        decompressed_data[:, :, unimportant_ids, :] = dequantized_unimportant_input
        decompressed_data[:, :, important_ids, :] = dequantized_important_input

    return decompressed_data

def true_mixedprec_channel_tokenwise_compress(
    input: torch.Tensor, important_ids, unimportant_ids, quantize_bit_important, quantize_bit_unimportant
):
    shape = input.shape
    # Extract important and unimportant tokens
    if len(important_ids.shape) == 3:
        important_indices_expanded = important_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        unimportant_indices_expanded = unimportant_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        important_data = torch.gather(input, 2, important_indices_expanded)
        unimportant_data = torch.gather(input, 2, unimportant_indices_expanded)        
    elif len(important_ids.shape) == 2:
        important_indices_expanded = important_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        unimportant_indices_expanded = unimportant_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        important_data = torch.gather(input, 2, important_indices_expanded)
        unimportant_data = torch.gather(input, 2, unimportant_indices_expanded)
    elif len(important_ids.shape) == 1:
        # This only works for 1-d ids
        important_data = input[:, :, important_ids, :]
        unimportant_data = input[:,:,unimportant_ids,:]

    quantized_important_data, min_important_data, step_important_data, \
         channel_max_important = true_channel_tokenwise_quantize(important_data, quantize_bit_important)
    
    quantized_unimportant_data, min_unimportant_data, step_unimportant_data, \
         channel_max_unimportant = true_channel_tokenwise_quantize(unimportant_data, quantize_bit_unimportant)

    return quantized_important_data, min_important_data, step_important_data, channel_max_important, \
        quantized_unimportant_data, min_unimportant_data, step_unimportant_data, channel_max_unimportant

def true_mixedprec_channel_tokenwise_decompress(quantized_important_data, min_important_data, step_important_data, quantize_bit_important, channel_max_important, \
        quantized_unimportant_data, min_unimportant_data, step_unimportant_data, quantize_bit_unimportant, channel_max_unimportant,\
            important_ids, unimportant_ids, dtype, shape):
    
    bsz, num_head, _, sep_dim = shape
    important_data_shape = torch.Size([bsz, num_head, quantized_important_data.shape[1], sep_dim])
    unimportant_data_shape = torch.Size([bsz, num_head, quantized_unimportant_data.shape[1], sep_dim])
    # Dequantize important data
    dequantized_important_input = tokenwise_dequantization_w_channelscale(
        quantized_important_data, quantize_bit_important, min_important_data, step_important_data, channel_max_important, important_data_shape, dtype
    )
    # dequantized_important_input = dequantized_important_input * channel_max_important

    # Dequantize unimportant data
    dequantized_unimportant_input = tokenwise_dequantization_w_channelscale(
        quantized_unimportant_data, quantize_bit_unimportant, min_unimportant_data, step_unimportant_data, channel_max_unimportant, unimportant_data_shape, dtype
    )
    # dequantized_unimportant_input = dequantized_unimportant_input * channel_max_unimportant

    # Merge them
    if len(important_ids.shape) == 3:
        decompressed_data = torch.zeros(shape, device=dequantized_important_input.device, dtype=dtype)
        important_indices_expanded = important_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1]) # (1,H,L,D)
        unimportant_indices_expanded = unimportant_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        decompressed_data = decompressed_data.scatter_(2, important_indices_expanded, dequantized_important_input)
        decompressed_data = decompressed_data.scatter_(2, unimportant_indices_expanded, dequantized_unimportant_input)        
    elif len(important_ids.shape) == 2:
        decompressed_data = torch.zeros(shape, device=dequantized_important_input.device, dtype=dtype)
        important_indices_expanded = important_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1]) # (1,H,L,D)
        unimportant_indices_expanded = unimportant_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        decompressed_data = decompressed_data.scatter_(2, important_indices_expanded, dequantized_important_input)
        decompressed_data = decompressed_data.scatter_(2, unimportant_indices_expanded, dequantized_unimportant_input)
    elif len(important_ids.shape) == 1:
        # This only works for 1-d ids
        decompressed_data[:, :, unimportant_ids, :] = dequantized_unimportant_input
        decompressed_data[:, :, important_ids, :] = dequantized_important_input

    return decompressed_data

def true_channel_separate_mixedprec_tokenwise_compress(
    input: torch.Tensor, important_ids, unimportant_ids, quantize_bit_important, quantize_bit_unimportant
):
    bsz, num_head, seq_len, sep_dim = input.shape
    shape = input.shape

    # Extract channel scales
    input = (
        input.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(shape[0], shape[2], shape[1] * shape[3])
    ) # bsz, seq_len, num_head*sep_dim
    channel_scale = torch.sqrt(torch.abs(input).reshape(-1, shape[1] * shape[3]).max(dim=0, keepdim=True).values.unsqueeze(0)) ## sqrt(max(abs))
    input = input / channel_scale
    input = (
        input.reshape(bsz, seq_len, num_head, sep_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    # Extract important and unimportant tokens
    if len(important_ids.shape) == 3:
        important_indices_expanded = important_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        unimportant_indices_expanded = unimportant_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        important_data = torch.gather(input, 2, important_indices_expanded)
        unimportant_data = torch.gather(input, 2, unimportant_indices_expanded)        
    elif len(important_ids.shape) == 2:
        important_indices_expanded = important_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        unimportant_indices_expanded = unimportant_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        important_data = torch.gather(input, 2, important_indices_expanded)
        unimportant_data = torch.gather(input, 2, unimportant_indices_expanded)
    elif len(important_ids.shape) == 1:
        # This only works for 1-d ids
        important_data = input[:, :, important_ids, :]
        unimportant_data = input[:,:,unimportant_ids,:]

    quantized_important_data, min_important_data, step_important_data = true_tokenwise_quantize(important_data, quantize_bit_important)
    
    quantized_unimportant_data, min_unimportant_data, step_unimportant_data = true_tokenwise_quantize(unimportant_data, quantize_bit_unimportant)

    return quantized_important_data, min_important_data, step_important_data, \
        quantized_unimportant_data, min_unimportant_data, step_unimportant_data, channel_scale

def true_channel_separate_mixedprec_tokenwise_decompress(quantized_important_data, min_important_data, step_important_data, quantize_bit_important, \
        quantized_unimportant_data, min_unimportant_data, step_unimportant_data, quantize_bit_unimportant, channel_scale,\
            important_ids, unimportant_ids, dtype, shape):
    
    bsz, num_head, _, sep_dim = shape
    important_data_shape = torch.Size([bsz, num_head, quantized_important_data.shape[1], sep_dim])
    unimportant_data_shape = torch.Size([bsz, num_head, quantized_unimportant_data.shape[1], sep_dim])
    # Dequantize important data
    dequantized_important_input = tokenwise_dequantization(
        quantized_important_data, quantize_bit_important, min_important_data, step_important_data, important_data_shape, dtype
    )

    # Dequantize unimportant data
    dequantized_unimportant_input = tokenwise_dequantization(
        quantized_unimportant_data, quantize_bit_unimportant, min_unimportant_data, step_unimportant_data, unimportant_data_shape, dtype
    )

    # Merge them
    if len(important_ids.shape) == 3:
        decompressed_data = torch.zeros(shape, device=dequantized_important_input.device, dtype=dtype)
        important_indices_expanded = important_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1]) # (1,H,L,D)
        unimportant_indices_expanded = unimportant_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        decompressed_data = decompressed_data.scatter_(2, important_indices_expanded, dequantized_important_input)
        decompressed_data = decompressed_data.scatter_(2, unimportant_indices_expanded, dequantized_unimportant_input)        
    elif len(important_ids.shape) == 2:
        decompressed_data = torch.zeros(shape, device=dequantized_important_input.device, dtype=dtype)
        important_indices_expanded = important_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1]) # (1,H,L,D)
        unimportant_indices_expanded = unimportant_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        decompressed_data = decompressed_data.scatter_(2, important_indices_expanded, dequantized_important_input)
        decompressed_data = decompressed_data.scatter_(2, unimportant_indices_expanded, dequantized_unimportant_input)
    elif len(important_ids.shape) == 1:
        # This only works for 1-d ids
        decompressed_data[:, :, unimportant_ids, :] = dequantized_unimportant_input
        decompressed_data[:, :, important_ids, :] = dequantized_important_input

    decompressed_data = (
        decompressed_data.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(shape[0], shape[2], shape[1] * shape[3])
    )
    decompressed_data *= channel_scale
    decompressed_data = (
        decompressed_data.reshape(bsz, -1, num_head, sep_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return decompressed_data

def true_mixedprec_tokenwise_compress(
    input: torch.Tensor, important_ids, unimportant_ids, quantize_bit_important, quantize_bit_unimportant
):
    shape = input.shape
    # Extract important and unimportant tokens
    if len(important_ids.shape) == 3:
        important_indices_expanded = important_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        unimportant_indices_expanded = unimportant_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        important_data = torch.gather(input, 2, important_indices_expanded)
        unimportant_data = torch.gather(input, 2, unimportant_indices_expanded)        
    elif len(important_ids.shape) == 2:
        important_indices_expanded = important_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        unimportant_indices_expanded = unimportant_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        important_data = torch.gather(input, 2, important_indices_expanded)
        unimportant_data = torch.gather(input, 2, unimportant_indices_expanded)
    elif len(important_ids.shape) == 1:
        # This only works for 1-d ids
        important_data = input[:, :, important_ids, :]
        unimportant_data = input[:,:,unimportant_ids,:]

    quantized_important_data, min_important_data, step_important_data = true_tokenwise_quantize(important_data, quantize_bit_important)
    
    quantized_unimportant_data, min_unimportant_data, step_unimportant_data = true_tokenwise_quantize(unimportant_data, quantize_bit_unimportant)

    return quantized_important_data, min_important_data, step_important_data, \
        quantized_unimportant_data, min_unimportant_data, step_unimportant_data

def true_mixedprec_tokenwise_decompress(quantized_important_data, min_important_data, step_important_data, quantize_bit_important, \
        quantized_unimportant_data, min_unimportant_data, step_unimportant_data, quantize_bit_unimportant,\
            important_ids, unimportant_ids, dtype, shape):
    
    bsz, num_head, _, sep_dim = shape
    important_data_shape = torch.Size([bsz, num_head, quantized_important_data.shape[1], sep_dim])
    unimportant_data_shape = torch.Size([bsz, num_head, quantized_unimportant_data.shape[1], sep_dim])
    # Dequantize important data
    dequantized_important_input = tokenwise_dequantization(
        quantized_important_data, quantize_bit_important, min_important_data, step_important_data, important_data_shape, dtype
    )

    # Dequantize unimportant data
    dequantized_unimportant_input = tokenwise_dequantization(
        quantized_unimportant_data, quantize_bit_unimportant, min_unimportant_data, step_unimportant_data, unimportant_data_shape, dtype
    )

    # Merge them
    if len(important_ids.shape) == 3:
        decompressed_data = torch.zeros(shape, device=dequantized_important_input.device, dtype=dtype)
        important_indices_expanded = important_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1]) # (1,H,L,D)
        unimportant_indices_expanded = unimportant_ids.unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        decompressed_data = decompressed_data.scatter_(2, important_indices_expanded, dequantized_important_input)
        decompressed_data = decompressed_data.scatter_(2, unimportant_indices_expanded, dequantized_unimportant_input)        
    elif len(important_ids.shape) == 2:
        decompressed_data = torch.zeros(shape, device=dequantized_important_input.device, dtype=dtype)
        important_indices_expanded = important_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1]) # (1,H,L,D)
        unimportant_indices_expanded = unimportant_ids.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, shape[-1])
        decompressed_data = decompressed_data.scatter_(2, important_indices_expanded, dequantized_important_input)
        decompressed_data = decompressed_data.scatter_(2, unimportant_indices_expanded, dequantized_unimportant_input)
    elif len(important_ids.shape) == 1:
        # This only works for 1-d ids
        decompressed_data[:, :, unimportant_ids, :] = dequantized_unimportant_input
        decompressed_data[:, :, important_ids, :] = dequantized_important_input

    return decompressed_data

def true_uniform_quantization_compress(input: torch.Tensor, quantize_bit):
    if quantize_bit != 8 and quantize_bit != 4:
        raise ValueError("quantize_bit should be 8 or 4")
    shape = input.shape
    bsz = shape[0]
    input = input.reshape(-1)
    if quantize_bit == 8:
        input = input.float()  # convert to 32bits to avoid max - min = inf
    min, max = input.min(), input.max()
    # step = (max - min) / (pow(2, quantize_bit) - 1)
    scale = (max - min) / (2**quantize_bit - 1)
    # print("before min max:",min,max,step)
    quantized_input = (input - min) / scale
    # print("after min max:",quantized_input.min(),quantized_input.max())
    # print("quantized isnan:",torch.any(torch.isnan(quantized_input)))
    quantized_input = quantized_input.round_()
    quantized_input = quantized_input.to(torch.uint8)
    if quantize_bit == 4:
        quantized_input = transfer_8bit_to_4bit(quantized_input)
    # print("isnan:",torch.any(torch.isnan(returning_input)))
    # while(True):
    #     pass
    return quantized_input, shape, min, scale


def true_uniform_quantization_decompress(
    input: torch.Tensor, quantize_bit, shape, min, step, dtype
):
    if quantize_bit != 8 and quantize_bit != 4:
        raise ValueError("quantize_bit should be 8 or 4")
    input = input.reshape(-1)
    if quantize_bit == 8:
        input = input.float()
        input = input * step + min
        output = input.reshape(shape).type(dtype)
    elif quantize_bit == 4:
        input = transfer_4bit_to_8bit(input)

        input = input.type(dtype)
        input = input * step + min
        output = input.reshape(shape)
    return output


def true_outlier_quantization_compress(input: torch.Tensor, quantize_bit, left):
    shape = input.shape
    input = input.reshape(-1)
    left_num = int(len(input) * left / 2)
    value1, indices1 = torch.topk(input, left_num, largest=False)
    value2, indices2 = torch.topk(input, left_num, largest=True)
    values = torch.cat((value1, value2), dim=0)
    indices = torch.cat((indices1, indices2), dim=0)

    input = input.index_fill_(0, indices, 0)
    output, _, min, step = true_uniform_quantization_compress(input, quantize_bit)

    return output, shape, min, step, values, indices


def true_outlier_quantization_decompress(
    input: torch.Tensor, quantize_bit, shape, min, step, dtype, values, indices
):
    input = true_uniform_quantization_decompress(
        input, quantize_bit, shape, min, step, dtype
    )
    input = input.reshape(-1)
    input[indices] = values
    input = input.reshape(shape)
    return input


def fake_quant_error_simulation(input: torch.Tensor, quantize_bit):
    input = input.reshape(-1)

    min, max = input.min(), input.max()
    step = (max - min) / (pow(2, quantize_bit) - 1)
    # print("before min max:",min,max,step)
    error = input - torch.round((input - min) / step)
    return error, min, step


def true_poweriteration(input: torch.Tensor, loop, rank, p_base=None, q_base=None):
    # input size [batch,num_head,seq_len,model_dim/num_head]
    # -> [batch,seq_len,model_dim] -> [batch * seq_len,model_dim]
    # p_base = torch.rand(input.shape[3] * input.shape[1], rank).to(device)
    # q_base = torch.rand(input.shape[0] * input.shape[2], rank).to(device)
    batch, num_head, seq_len, sep_dim = input.shape
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )  # convert to 32bits for qr decomposition
    input = input.view(batch, seq_len, sep_dim * num_head)
    input = input.float()
    if q_base is not None and p_base is not None:
        p_base[0] = p_base[0].float()
        q_base[0] = q_base[0].float()
    else:
        p_base = [torch.rand(batch, sep_dim * num_head, rank).to(input.device).float()]
        q_base = [torch.rand(batch, seq_len, rank).to(input.device).float()]
    # 3 calculation = loop * (matmul) + 2 * qrO(n^2)
    for i in range(loop):
        if i == loop - 1:
            p_base[0] = torch.linalg.qr(p_base[0].float()).Q
        q_base[0] = input @ p_base[0]
        if i == loop - 1:
            q_base[0] = torch.linalg.qr(q_base[0].float()).Q
        p_base[0] = torch.transpose(input, 1, 2) @ q_base[0]
    # input = q_base[0] @ torch.transpose(p_base[0], 0, 1)
    # input = input.view(batch, seq_len, num_head, sep_dim)
    # input = input.permute(0, 2, 1, 3)
    # input = input.type(torch.bfloat16)
    p_base[0] = p_base[0].half()
    q_base[0] = q_base[0].half()
    return p_base, q_base


def true_poweriteration_quantized(
    input: torch.Tensor, loop, rank, p_base=None, q_base=None
):
    # input size [batch,num_head,seq_len,model_dim/num_head]
    # -> [batch,seq_len,model_dim] -> [batch * seq_len,model_dim]
    # p_base = torch.rand(input.shape[3] * input.shape[1], rank).to(device)
    # q_base = torch.rand(input.shape[0] * input.shape[2], rank).to(device)
    batch, num_head, seq_len, sep_dim = input.shape
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )  # convert to 32bits for qr decomposition
    input = input.view(batch, seq_len, sep_dim * num_head)
    input = input.float()
    if q_base is not None and p_base is not None:
        p_base[0] = p_base[0].float()
        q_base[0] = q_base[0].float()
    else:
        p_base = [torch.rand(batch, sep_dim * num_head, rank).to(input.device).float()]
        q_base = [torch.rand(batch, seq_len, rank).to(input.device).float()]
    # 3 calculation = loop * (matmul) + 2 * qrO(n^2)
    for i in range(loop):
        if i == loop - 1:
            p_base[0] = torch.linalg.qr(p_base[0].float()).Q
        q_base[0] = input @ p_base[0]
        if i == loop - 1:
            q_base[0] = torch.linalg.qr(q_base[0].float()).Q
        p_base[0] = torch.transpose(input, 1, 2) @ q_base[0]
    # input = q_base[0] @ torch.transpose(p_base[0], 0, 1)
    # input = input.view(batch, seq_len, num_head, sep_dim)
    # input = input.permute(0, 2, 1, 3)
    # input = input.type(torch.bfloat16)

    # p_base[0] = p_base[0].half()
    # q_base[0] = q_base[0].half()
    #### compress p and q base to 8bits
    p_base[0], shape_p, min_p, scale_p = true_uniform_quantization_compress(
        p_base[0], 8
    )
    q_base[0], shape_q, min_q, scale_q = true_uniform_quantization_compress(
        q_base[0], 8
    )
    return p_base, q_base, shape_p, shape_q, min_p, min_q, scale_p, scale_q


def true_gear_compress(input: torch.Tensor, quantize_bit, left, rank, loop):
    shape = input.shape
    input = input.reshape(-1)
    left_num = int(len(input) * left / 2)
    value1, indices1 = torch.topk(input, left_num, largest=False)
    value2, indices2 = torch.topk(input, left_num, largest=True)
    values = torch.cat((value1, value2), dim=0)
    indices = torch.cat((indices1, indices2), dim=0)
    input = input.index_fill_(0, indices, 0)
    error, min, step = fake_quant_error_simulation(input, quantize_bit)
    error = error.index_fill_(0, indices, 0)
    error = error.reshape(shape)
    p_base, q_base = true_poweriteration(error, loop, rank)
    # has_inf = torch.isinf(p_base[0])
    # has_nan = torch.isnan(p_base[0])
    # if has_inf.any() or has_nan.any():
    #     print("pbase",has_inf.any(),has_nan.any())
    # has_inf = torch.isinf(q_base[0])
    # has_nan = torch.isnan(q_base[0])
    # if has_inf.any() or has_nan.any():
    #     print("qbase",has_inf.any(),has_nan.any())
    output, _, min, step = true_uniform_quantization_compress(input, quantize_bit)
    return output, shape, min, step, values, indices, p_base, q_base


def true_gear_decompress(
    input: torch.Tensor,
    quantize_bit,
    shape,
    min,
    step,
    dtype,
    values,
    indices,
    p_base,
    q_base,
):

    input = true_uniform_quantization_decompress(
        input, quantize_bit, shape, min, step, dtype
    )
    input = input.reshape(-1)
    input[indices] = values
    input = input.reshape(shape)
    error = q_base[0] @ torch.transpose(p_base[0], 1, 2)
    batch, num_head, seq_len, sep_dim = input.shape
    error = error.reshape(batch, seq_len, num_head, sep_dim)
    # error = error.permute(0, 2, 1, 3).type(input.dtype)
    error = error.permute(0, 2, 1, 3)
    input = input + error

    return input


def true_uniform_quantization_compress_batchwise(input: torch.Tensor, quantize_bit):

    if quantize_bit != 8 and quantize_bit != 4:
        raise ValueError("quantize_bit should be 8 or 4")
    shape = input.shape
    bsz = shape[0]
    input = input.reshape(bsz, -1)
    if quantize_bit == 8:
        input = input.float()  # convert to 32bits to avoid max - min = inf
    min, max = input.min(dim=-1).values, input.max(dim=-1).values
    step = (max - min) / (pow(2, quantize_bit) - 1)
    min = min.unsqueeze(1)  # Expand min tensor shape to (bsz, 1)
    step = step.unsqueeze(1)  # Expand step tensor shape to (bsz, 1)
    # print("before min max:",min,max,step)
    input = torch.round((input - min) / step)
    # print("after min max:",quantized_input.min(),quantized_input.max())
    # print("quantized isnan:",torch.any(torch.isnan(quantized_input)))
    input = input.to(torch.uint8)
    if quantize_bit == 4:
        input = transfer_8bit_to_4bit_batchwise(input)
    # print("isnan:",torch.any(torch.isnan(returning_input)))
    # while(True):
    #     pass
    return input, shape, min, step


def true_uniform_quantization_decompress_batchwise(
    input: torch.Tensor, quantize_bit, shape, min, step, dtype
):
    if quantize_bit != 8 and quantize_bit != 4:
        raise ValueError("quantize_bit should be 8 or 4")

    bsz = shape[0]
    input = input.reshape(bsz, -1)
    if quantize_bit == 8:
        input = input.float()
        input = input * step + min

        output = input.reshape(shape).type(dtype)
    elif quantize_bit == 4:
        input = transfer_4bit_to_8bit_batchwise(input)

        input = input.type(dtype)
        input = input * step + min
        output = input.reshape(shape)

    return output


def true_outlier_quantization_compress_batchwise(
    input: torch.Tensor, quantize_bit, left
):
    shape = input.shape
    bsz = shape[0]
    input = input.reshape(bsz, -1)
    left_num = int(input.numel() / bsz * left / 2)

    value1, indices1 = torch.topk(input, left_num, largest=False, dim=-1)
    value2, indices2 = torch.topk(input, left_num, largest=True, dim=-1)

    values = torch.cat((value1, value2), dim=-1)
    indices = torch.cat((indices1, indices2), dim=-1)
    # input = input.index_fill_(0,indices,0)
    # print(indices.shape)
    input.scatter_(1, indices, 0)

    output, _, min, step = true_uniform_quantization_compress(input, quantize_bit)

    return output, shape, min, step, values, indices

def true_outlier_quantization_decompress_batchwise(
    input: torch.Tensor, quantize_bit, shape, min, step, dtype, values, indices
):
    bsz = shape[0]
    input = true_uniform_quantization_decompress(
        input, quantize_bit, shape, min, step, dtype
    )
    input = input.reshape(bsz, -1)
    input.scatter_(1, indices, values)
    input = input.reshape(shape)
    return input


def fake_quant_error_simulation_batchwise(input: torch.Tensor, quantize_bit, bsz):
    input = input.reshape(bsz, -1)

    min, max = input.min(dim=-1).values, input.max(dim=-1).values

    step = (max - min) / (pow(2, quantize_bit) - 1)
    min = min.unsqueeze(1)  # Expand min tensor shape to (bsz, 1)
    step = step.unsqueeze(1)  # Expand step tensor shape to (bsz, 1)
    # print("before min max:",min,max,step)
    error = input - (torch.round((input - min) / step) * step + min)
    return error, min, step


def true_gear_compress_batchwise(input: torch.Tensor, quantize_bit, left, rank, loop):
    shape = input.shape
    bsz = shape[0]
    input = input.reshape(bsz, -1)
    left_num = int(input.numel() / bsz * left / 2)
    value1, indices1 = torch.topk(input, left_num, largest=False, dim=-1)
    value2, indices2 = torch.topk(input, left_num, largest=True, dim=-1)
    values = torch.cat((value1, value2), dim=-1)
    indices = torch.cat((indices1, indices2), dim=-1)
    input = input.scatter_(1, indices, 0.0)
    error, min, step = fake_quant_error_simulation_batchwise(input, quantize_bit, bsz)
    error = error.scatter_(1, indices, 0.0)
    error = error.reshape(shape)
    bsz, num_head, seq_len, sep_dim = shape
    smaller_dim = seq_len if seq_len < sep_dim * num_head else sep_dim * num_head
    rank = int(rank * smaller_dim)
    p_base, q_base = true_poweriteration(error, loop, rank)
    # has_inf = torch.isinf(p_base[0])
    # has_nan = torch.isnan(p_base[0])
    # if has_inf.any() or has_nan.any():
    #     print("pbase",has_inf.any(),has_nan.any())
    # has_inf = torch.isinf(q_base[0])
    # has_nan = torch.isnan(q_base[0])
    # if has_inf.any() or has_nan.any():
    #     print("qbase",has_inf.any(),has_nan.any())
    output, _, min, step = true_uniform_quantization_compress(input, quantize_bit)
    return output, shape, min, step, values, indices, p_base, q_base


def true_gear_decompress_batchwise(
    input: torch.Tensor,
    quantize_bit,
    shape,
    min,
    step,
    dtype,
    values,
    indices,
    p_base,
    q_base,
):
    bsz = shape[0]
    input = true_uniform_quantization_decompress(
        input, quantize_bit, shape, min, step, dtype
    )
    input = input.reshape(bsz, -1)
    input.scatter_(1, indices, values)
    input = input.reshape(shape)
    error = q_base[0] @ torch.transpose(p_base[0], 1, 2)
    batch, num_head, seq_len, sep_dim = input.shape
    error = error.reshape(batch, seq_len, num_head, sep_dim)
    # error = error.permute(0, 2, 1, 3).type(input.dtype)
    error = error.permute(0, 2, 1, 3)
    input = input + error

    return input


def tokenwise_quantization_compress_with_error(input: torch.Tensor, quantize_bit):
    # # Currently only support 4 bit quantization
    # assert quantize_bit == 4
    shape = input.shape  # bsz, num_head, seq_len, sep_dim
    input = (
        input.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(shape[0], shape[2], shape[1] * shape[3])
    ) # bsz, seq_len, num_head*sep_dim
    min, max = input.min(dim=-1).values.unsqueeze(-1), input.max(
        dim=-1
    ).values.unsqueeze(-1)
    step = (max - min) / (pow(2, quantize_bit) - 1)

    quantized_input = (input - min) / step
    # quantized_input = F.relu(quantized_input)
    quantized_input = quantized_input.round_()
    error = input - (quantized_input * step + min)

    quantized_input = quantized_input.to(torch.uint8)
    if quantize_bit == 4:
        quantized_input = transfer_8bit_to_4bit_batchwise(quantized_input)
    elif quantize_bit == 2:
        quantized_input = transfer_8bit_to_2bit_batchwise(quantized_input)

    # reshape back to original shape
    # quantized_input = quantized_input.reshape(shape[0],shape[2],shape[1],shape[3])
    error = error.reshape(shape[0], shape[2], shape[1], shape[3])
    # quantized_input = quantized_input.permute(0, 2, 1, 3).contiguous()
    error = error.permute(0, 2, 1, 3).contiguous()
    return quantized_input, error, min, step, shape


def tokenwise_dequantization(
    quantized_input: torch.Tensor, quantize_bit, min, step, shape, dtype
):
    # input size bsz, seq_len, -1
    # assert quantize_bit == 4 or quantize_bit == 8
    if quantize_bit == 4:
        quantized_input = transfer_4bit_to_8bit_batchwise(quantized_input)
    elif quantize_bit == 2:
        quantized_input = transfer_2bit_to_8bit_batchwise(quantized_input)

    quantized_input = quantized_input.to(dtype)
    bsz, num_head, seq_len, sep_dim = shape
    quantized_input = quantized_input * step + min
    quantized_input = (
        quantized_input.reshape(bsz, seq_len, num_head, sep_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    return quantized_input

def tokenwise_dequantization_w_channelscale(
    quantized_input: torch.Tensor, quantize_bit, min, step, channel_scale, shape, dtype
):
    # input size bsz, seq_len, -1
    # assert quantize_bit == 4 or quantize_bit == 8
    if quantize_bit == 4:
        quantized_input = transfer_4bit_to_8bit_batchwise(quantized_input)
    elif quantize_bit == 2:
        quantized_input = transfer_2bit_to_8bit_batchwise(quantized_input)

    quantized_input = quantized_input.to(dtype)
    bsz, num_head, seq_len, sep_dim = shape
    quantized_input = quantized_input * step + min
    quantized_input *= channel_scale
    quantized_input = (
        quantized_input.reshape(bsz, seq_len, num_head, sep_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    return quantized_input

def true_gear_tokenwiseQ_compress(input: torch.Tensor, quantize_bit, rank, loop):
    shape = input.shape  # bsz, num_head, seq_len, sep_dim
    bsz = shape[0]
    quantized_input, error, min, step, shape = (
        tokenwise_quantization_compress_with_error(input, quantize_bit)
    )
    # print("min_max_error_compress:",error.min(),error.max())
    bsz, num_head, seq_len, sep_dim = shape
    smaller_dim = seq_len if seq_len < sep_dim * num_head else sep_dim * num_head
    rank = int(rank * smaller_dim)
    p_base, q_base, shape_p, shape_q, min_p, min_q, scale_p, scale_q = (
        true_poweriteration_quantized(error, loop, rank)
    )
    del error
    return (
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
    )


def true_gear_tokenwiseQ_decompress(
    quantized_input,
    quantize_bit,
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
    dtype,
):
    # bsz = shape[0]
    #### TODO

    dequantized_input = tokenwise_dequantization(
        quantized_input, quantize_bit, min, step, shape, dtype
    )
    p_base_dequantized = true_uniform_quantization_decompress(
        p_base[0], 8, shape_p, min_p, scale_p, torch.float
    )
    q_base_dequantized = true_uniform_quantization_decompress(
        q_base[0], 8, shape_q, min_q, scale_q, torch.float
    )
    error = q_base_dequantized @ torch.transpose(p_base_dequantized, 1, 2)
    batch, num_head, seq_len, sep_dim = dequantized_input.shape
    error = error.reshape(batch, seq_len, num_head, sep_dim)
    error = error.permute(0, 2, 1, 3)
    # print("min_max_error_decompress:",error.min(),error.max())
    dequantized_input = dequantized_input + error.to(dtype)
    return dequantized_input


#### no pq quantization


def true_gear_tokenwiseQ_compress_nopq(input: torch.Tensor, quantize_bit, rank, loop):
    shape = input.shape  # bsz, num_head, seq_len, sep_dim
    bsz = shape[0]
    quantized_input, error, min, step, shape = (
        tokenwise_quantization_compress_with_error(input, quantize_bit)
    )
    # print("min_max_error_compress:",error.min(),error.max())
    bsz, num_head, seq_len, sep_dim = shape
    smaller_dim = seq_len if seq_len < sep_dim * num_head else sep_dim * num_head
    rank = int(rank * smaller_dim)
    p_base, q_base = true_poweriteration(error, loop, rank)
    del error
    return quantized_input, shape, min, step, p_base, q_base

def true_gear_tokenwiseQ_decompress_nopq(
    quantized_input, quantize_bit, shape, min, step, p_base, q_base, dtype
):
    # bsz = shape[0]
    #### TODO

    dequantized_input = tokenwise_dequantization(
        quantized_input, quantize_bit, min, step, shape, dtype
    )

    error = q_base[0] @ torch.transpose(p_base[0], 1, 2)
    batch, num_head, seq_len, sep_dim = dequantized_input.shape
    error = error.reshape(batch, seq_len, num_head, sep_dim)
    error = error.permute(0, 2, 1, 3)
    # print("min_max_error_decompress:",error.min(),error.max())
    dequantized_input = dequantized_input + error.to(dtype)
    return dequantized_input


def true_gear_outlier_tokenwiseQ_compress_nopq(input: torch.Tensor, quantize_bit, left, rank, loop):
    shape = input.shape  # bsz, num_head, seq_len, sep_dim
    bsz = shape[0]

    # Extract outliers
    input = input.reshape(bsz, -1)
    left_num = int(input.numel() / bsz * left / 2)
    value1, indices1 = torch.topk(input, left_num, largest=False, dim=-1)
    value2, indices2 = torch.topk(input, left_num, largest=True, dim=-1)
    values = torch.cat((value1, value2), dim=-1)
    indices = torch.cat((indices1, indices2), dim=-1)
    input = input.scatter_(1, indices, 0.0)
    input = input.reshape(shape)

    quantized_input, error, min, step, shape = (
        tokenwise_quantization_compress_with_error(input, quantize_bit)
    )
    # print("min_max_error_compress:",error.min(),error.max())
    bsz, num_head, seq_len, sep_dim = shape
    smaller_dim = seq_len if seq_len < sep_dim * num_head else sep_dim * num_head
    rank = int(rank * smaller_dim)
    p_base, q_base = true_poweriteration(error, loop, rank)
    del error
    return quantized_input, shape, min, step, values, indices, p_base, q_base

def true_gear_outlier_tokenwiseQ_decompress_nopq(
    quantized_input, quantize_bit, shape, min, step, values, indices, p_base, q_base, dtype
):
    bsz = shape[0]

    dequantized_input = tokenwise_dequantization(
        quantized_input, quantize_bit, min, step, shape, dtype
    )

    dequantized_input = dequantized_input.reshape(bsz, -1)
    dequantized_input.scatter_(1, indices, values)
    dequantized_input = dequantized_input.reshape(shape)

    error = q_base[0] @ torch.transpose(p_base[0], 1, 2)
    batch, num_head, seq_len, sep_dim = dequantized_input.shape
    error = error.reshape(batch, seq_len, num_head, sep_dim)
    error = error.permute(0, 2, 1, 3)
    # print("min_max_error_decompress:",error.min(),error.max())
    dequantized_input = dequantized_input + error.to(dtype)
    return dequantized_input

def true_channel_tokenwise_quantize(input, quantize_bit):
    shape = input.shape  # bsz, num_head, seq_len, sep_dim
    
    input = (
        input.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(shape[0], shape[2], shape[1] * shape[3])
    ) # bsz, seq_len, num_head*sep_dim

    # if quantize_bit == 8:
    #     channel_scale = 1
    # else:
    # min = input.reshape(-1, shape[1] * shape[3]).min(dim=0, keepdim=True).values.unsqueeze(0)
    # max = input.reshape(-1, shape[1] * shape[3]).max(dim=0, keepdim=True).values.unsqueeze(0)
    # channel_scale = torch.sqrt(max-min)
    channel_scale = torch.sqrt(torch.abs(input).reshape(-1, shape[1] * shape[3]).max(dim=0, keepdim=True).values.unsqueeze(0)) ## sqrt(max(abs))
    # channel_scale = torch.abs(input).reshape(-1, shape[1] * shape[3]).max(dim=0, keepdim=True).values.unsqueeze(0) ## max(abs)
    # channel_scale = torch.sqrt(torch.mean(torch.abs(input), dim=(0,1)).unsqueeze(0).unsqueeze(0)) ## sqrt(mean(abs))
    # mean_scale = torch.mean(torch.abs(input), dim=(0,1)).unsqueeze(0).unsqueeze(0) * 7.5 ## mean(abs())
    # import scipy.stats as st
    # print(st.linregress(channel_scale.cpu().flatten().numpy(), mean_scale.cpu().flatten().numpy()))

    # channel_scale = 1
    input = input / channel_scale

    min, max = input.min(dim=-1).values.unsqueeze(-1), input.max(
        dim=-1
    ).values.unsqueeze(-1)
    step = (max - min) / (pow(2, quantize_bit) - 1)

    quantized_input = (input - min) / step
    # quantized_input = F.relu(quantized_input)
    quantized_input = quantized_input.round_()
    quantized_input = quantized_input.to(torch.uint8)

    if quantize_bit == 4:
        quantized_input = transfer_8bit_to_4bit_batchwise(quantized_input)
    elif quantize_bit == 2:
        quantized_input = transfer_8bit_to_2bit_batchwise(quantized_input)

    return quantized_input, min, step, channel_scale

def true_tokenwise_quantize(input, quantize_bit):
    shape = input.shape  # bsz, num_head, seq_len, sep_dim
    
    input = (
        input.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(shape[0], shape[2], shape[1] * shape[3])
    ) # bsz, seq_len, num_head*sep_dim

    min, max = input.min(dim=-1).values.unsqueeze(-1), input.max(
        dim=-1
    ).values.unsqueeze(-1)
    step = (max - min) / (pow(2, quantize_bit) - 1)

    quantized_input = (input - min) / step
    # quantized_input = F.relu(quantized_input)
    quantized_input = quantized_input.round_()
    quantized_input = quantized_input.to(torch.uint8)

    if quantize_bit == 4:
        quantized_input = transfer_8bit_to_4bit_batchwise(quantized_input)
    elif quantize_bit == 2:
        quantized_input = transfer_8bit_to_2bit_batchwise(quantized_input)

    return quantized_input, min, step

def true_groupwise_quantize(input, quantize_bit, group_size=32):
    bsz, num_head, seq_len, sep_dim = input.shape  # bsz, num_head, seq_len, sep_dim
    new_shape = (bsz*num_head*seq_len*sep_dim//group_size, group_size)

    input = input.reshape(new_shape)

    min, max = input.min(dim=-1).values.unsqueeze(-1), input.max(
        dim=-1
    ).values.unsqueeze(-1)
    step = (max - min) / (pow(2, quantize_bit) - 1)

    quantized_input = (input - min) / step
    # quantized_input = F.relu(quantized_input)
    quantized_input = quantized_input.round_()
    quantized_input = quantized_input.to(torch.uint8)

    if quantize_bit == 4:
        quantized_input = transfer_8bit_to_4bit_batchwise(quantized_input)
    elif quantize_bit == 2:
        quantized_input = transfer_8bit_to_2bit_batchwise(quantized_input)

    return quantized_input, min, step