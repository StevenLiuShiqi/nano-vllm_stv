from nanovllm.engine.block_manager import BlockManager          
from nanovllm.engine.sequence import Sequence                     
                                              
bm = BlockManager(num_blocks=20, block_size=4)                    
                    
# First request: 12 token, using 3 blocks                      
seq_a = Sequence([1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12])
seq_a.block_size = 4
bm.allocate(seq_a)                                                
print("After seq_a allocate:", bm.stats())                      
print("seq_a block_table:", seq_a.block_table)                    
print("seq_a num_cached_tokens:", seq_a.num_cached_tokens)        
                                                                  
# Second request: first 8 tokens identical with the first req
seq_b = Sequence([1, 2, 3, 4,  5, 6, 7, 8,  99, 100, 101, 102])
seq_b.block_size = 4
bm.allocate(seq_b)                                                
print("\nAfter seq_b allocate:", bm.stats()) 
print("seq_b block_table:", seq_b.block_table)                    
print("seq_b num_cached_tokens:", seq_b.num_cached_tokens)        
                                                                
# verification that the 2nd req is sharing blocks with the 1st one                                    
print("\nShared blocks?", seq_a.block_table[:2] ==
seq_b.block_table[:2])                                            
                                              
# ref_count                                      
for bid in seq_a.block_table:
    b = bm.blocks[bid]                                            
    print(f"  block {bid}: ref_count={b.ref_count}, hash={b.hash != -1}")                     