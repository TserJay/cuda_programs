

# 5060ti


input numel=25600000 dtype=torch.float32 device=cuda:0
reduce_1 naive           out=25600000.00000000 time=0.51442725 ms
reduce_1 naive           correct=True
reduce_2 grid_stride     out=25600000.00000000 time=0.25533311 ms
reduce_2 grid_stride     correct=True
reduce_3 warp_atomic     out=25600000.00000000 time=0.25029327 ms
reduce_3 warp_atomic     correct=True


input numel=25600000 dtype=torch.float32 device=cuda:0
reduce_1 naive           out=25600000.00000000 time=0.51462769 ms
reduce_1 naive           correct=True
reduce_2 grid_stride     out=25600000.00000000 time=0.54604657 ms
reduce_2 grid_stride     correct=True
reduce_3 warp_atomic     out=25600000.00000000 time=0.24613197 ms
reduce_3 warp_atomic     correct=True





