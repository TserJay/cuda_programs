

# 5060ti

1. 
input numel=25600000 dtype=torch.float32 device=cuda:0
reduce_1 naive           out=25600000.00000000 time=0.51442725 ms
reduce_1 naive           correct=True
reduce_2 grid_stride     out=25600000.00000000 time=0.25533311 ms
reduce_2 grid_stride     correct=True
reduce_3 warp_atomic     out=25600000.00000000 time=0.25029327 ms
reduce_3 warp_atomic     correct=True


2. 
input numel=25600000 dtype=torch.float32 device=cuda:0
reduce_1 naive           out=25600000.00000000 time=0.51462769 ms
reduce_1 naive           correct=True
reduce_2 grid_stride     out=25600000.00000000 time=0.54604657 ms
reduce_2 grid_stride     correct=True
reduce_3 warp_atomic     out=25600000.00000000 time=0.24613197 ms
reduce_3 warp_atomic     correct=True


3. 2026040414 
reduce_11 warp divergence 出现了性能的下降

input numel=25600000 dtype=torch.float32 device=cuda:0
reduce_1 naive                 out=25600000.00000000 time=0.50846561 ms
reduce_1 naive                 correct=True
reduce_11 warp divergence      out=25600000.00000000 time=0.54178143 ms
reduce_11 warp divergence      correct=True
reduce_2                       out=25600000.00000000 time=0.50431226 ms
reduce_2                       correct=True
reduce_101 grid_stride         out=25600000.00000000 time=0.25119711 ms
reduce_101 grid_stride         correct=True
reduce_102 warp_atomic         out=25600000.00000000 time=0.25368352 ms
reduce_102 warp_atomic         correct=True


4. 2026040416
input numel=25600000 dtype=torch.float32 device=cuda:0
reduce_1 naive                 out=25600000.00000000 time=0.51333795 ms
reduce_1 naive                 correct=True
reduce_11 warp divergence      out=25600000.00000000 time=0.54345135 ms
reduce_11 warp divergence      correct=True
reduce_2                       out=25600000.00000000 time=0.50438351 ms
reduce_2                       correct=True
reduce_3                       out=25600000.00000000 time=0.27064819 ms
reduce_3                       correct=True
reduce_4                       out=25600000.00000000 time=0.25792059 ms
reduce_4                       correct=True
reduce_5                       out=25600000.00000000 time=0.25801456 ms
reduce_5                       correct=True
reduce_101 grid_stride         out=25600000.00000000 time=0.25120813 ms
reduce_101 grid_stride         correct=True
reduce_102 warp_atomic         out=25600000.00000000 time=0.24886336 ms
reduce_102 warp_atomic         correct=True


5. 2026040418
reduce_1 naive                 out=25600000.00000000 time=0.51615137 ms
reduce_1 naive                 correct=True
reduce_11 warp divergence      out=25600000.00000000 time=0.55360486 ms
reduce_11 warp divergence      correct=True
reduce_2                       out=25600000.00000000 time=0.51080722 ms
reduce_2                       correct=True
reduce_3                       out=25600000.00000000 time=0.27780035 ms
reduce_3                       correct=True
reduce_4                       out=25600000.00000000 time=0.26184964 ms
reduce_4                       correct=True
reduce_5                       out=25600000.00000000 time=0.26214740 ms
reduce_5                       correct=True
reduce_6                       out=25600000.00000000 time=0.26094681 ms
reduce_6                       correct=True
reduce_6                       out=25600000.00000000 time=0.25616727 ms ---更少的gridsize
reduce_6                       correct=True
reduce_101 grid_stride         out=25600000.00000000 time=0.25452499 ms
reduce_101 grid_stride         correct=True
reduce_102 warp_atomic         out=25600000.00000000 time=0.25040643 ms
reduce_102 warp_atomic         correct=True


6. 2026040418
reduce_1 naive                 out=25600000.00000000 time=0.51700488 ms
reduce_11 warp divergence      out=25600000.00000000 time=0.54903241 ms
reduce_2                       out=25600000.00000000 time=0.51681451 ms
reduce_3                       out=25600000.00000000 time=0.27497711 ms
reduce_4                       out=25600000.00000000 time=0.26192856 ms
reduce_5                       out=25600000.00000000 time=0.26293094 ms
reduce_6                       out=25600000.00000000 time=0.26713239 ms
reduce_6                       out=25600000.00000000 time=0.25625275 ms ---更少的gridsize
reduce_7                       out=25600000.00000000 time=0.26432721 ms
reduce_7                       out=25600000.00000000 time=0.25602838 ms ---更少的gridsize
reduce_601                     out=25600000.00000000 time=0.26579706 ms
reduce_701                     out=25600000.00000000 time=0.26640573 ms
reduce_101                     out=25600000.00000000 time=0.25479005 ms
reduce_102 warp_atomic         out=25600000.00000000 time=0.25259216 ms















# A100

显卡 0 负载
1. 2026040414 
input numel=25600000 dtype=torch.float32 device=cuda:0
reduce_1 naive           out=100000.00000000 time=0.18287514 ms  ---位运算：if (tid & (2 * index - 1) == 0)
reduce_1 naive           correct=False
reduce_11 warp divergence out=25600000.00000000 time=0.28222055 ms
reduce_11 warp divergence correct=True
reduce_2 grid_stride     out=25600000.00000000 time=0.08474829 ms
reduce_2 grid_stride     correct=True
reduce_3 warp_atomic     out=25600000.00000000 time=0.07777997 ms
reduce_3 warp_atomic     correct=True

2. 2026040414 
input numel=25600000 dtype=torch.float32 device=cuda:0
reduce_1 naive           out=25600000.00000000 time=0.51381763 ms ---模（除）：if (tid % (2 * index) == 0) 
reduce_1 naive           correct=True
reduce_11 warp divergence out=25600000.00000000 time=0.28273459 ms
reduce_11 warp divergence correct=True
reduce_101 grid_stride   out=25600000.00000000 time=0.08489677 ms
reduce_101 grid_stride   correct=True
reduce_102 warp_atomic   out=25600000.00000000 time=0.07795200 ms
reduce_102 warp_atomic   correct=True

3. 2026040414 
input numel=25600000 dtype=torch.float32 device=cuda:0
reduce_1 naive                 out=25600000.00000000 time=0.51370087 ms
reduce_1 naive                 correct=True
reduce_11 warp divergence      out=25600000.00000000 time=0.28242432 ms
reduce_11 warp divergence      correct=True
reduce_2                       out=25600000.00000000 time=0.22283775 ms
reduce_2                       correct=True
reduce_101 grid_stride         out=25600000.00000000 time=0.08489370 ms
reduce_101 grid_stride         correct=True
reduce_102 warp_atomic         out=25600000.00000000 time=0.07785165 ms
reduce_102 warp_atomic         correct=True


4. 2026040416(非0负载)
input numel=25600000 dtype=torch.float32 device=cuda:1
reduce_1 naive                 out=25600000.00000000 time=0.56863641 ms
reduce_1 naive                 correct=True
reduce_11 warp divergence      out=25600000.00000000 time=0.33868390 ms
reduce_11 warp divergence      correct=True
reduce_2                       out=25600000.00000000 time=0.22578073 ms
reduce_2                       correct=True
reduce_3                       out=25600000.00000000 time=0.16607744 ms
reduce_3                       correct=True
reduce_4                       out=25600000.00000000 time=0.09811353 ms
reduce_4                       correct=True
reduce_101 grid_stride         out=25600000.00000000 time=0.08471449 ms
reduce_101 grid_stride         correct=True
reduce_102 warp_atomic         out=25600000.00000000 time=0.07783731 ms
reduce_102 warp_atomic         correct=True



5. 2026040416 
input numel=25600000 dtype=torch.float32 device=cuda:1
reduce_1 naive                 out=25600000.00000000 time=0.51318579 ms
reduce_1 naive                 correct=True
reduce_11 warp divergence      out=25600000.00000000 time=0.28231989 ms
reduce_11 warp divergence      correct=True
reduce_2                       out=25600000.00000000 time=0.22280397 ms
reduce_2                       correct=True
reduce_3                       out=25600000.00000000 time=0.12324147 ms
reduce_3                       correct=True
reduce_4                       out=25600000.00000000 time=0.09817088 ms
reduce_4                       correct=True
reduce_5                       out=25600000.00000000 time=0.09800294 ms
reduce_5                       correct=True
reduce_101 grid_stride         out=25600000.00000000 time=0.08474419 ms
reduce_101 grid_stride         correct=True
reduce_102 warp_atomic         out=25600000.00000000 time=0.07789465 ms
reduce_102 warp_atomic         correct=True



