# 1
## a

128 / 32 = 4
## b
((1024+128-1)/128) * 4 = 32
## c
### i
warps: [0,32), [32, 64), [96, 128) ... so total 31 warps
### ii
[32, 64), [96, 128)  2 warps are divergent
### iii
warp 0 of block 0 is [0, 32), 100%
### iv
warp 1 of block 0 is [32, 64), [32, 40) is active, [40, 64) is inactive, so the effciency is 25%
### v
warp 3 of block 0 is [96, 128), [96, 104) is inactive, [104, 128) is active, so the effiency is 75%
## d
### i
32
### ii
32
### iii
50%
## e
`5 - (i%3)` 's range is [2, 5], so total 5 iteration, the 0, and 1 have no divergence, the 2, 3, 4  have divergence
### i
2
### ii
3
# 2
2,048
# 3
just one [1,984, 2016), over the 2000 index.
# 4
the total execution time depends on the longest one, 3.0ns.
1+0.7+0+0.2+0.6+1.1+0.4+0.1=4.1ns
3 * 8 = 24ns
4.1/24=0.1708333333
%17%

# 5
no, the execution time of thread is no determined by the number of threads in block.

# 6
c

# 7
a. 50%

b. 50%

c. 50%

d. 100%

e. 100%

# 8
## a
128 threads/block * 32 blocks/SM = 4096 threads/SM > limit(2048 threads/SM)
## b
no
## c
256 threads/block * 32 blocks/SM = 8,192 threads/SM > limit(2048 threads/SM)
34 registers/thread * 2048  threads/SM = 69,632 registers/SM > limit(65536 registers/SM)
# 9

32 * 32 = 1024 threads/block is bigger than the limit of CUDA device that allows up to 512 threads/block.
I suggest him to reduce the blocks size, and add more blocks to deal the martix mulpilication.

