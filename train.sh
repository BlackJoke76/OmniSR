CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 1 --master_port 29500 ./train_DDP.py 