## Config Difference

| parameter name          | official V1      | VCTK_cpc100      | VCTK_VQVAE       | VCTK_HuBERT      |
| ----------------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| "upsample_rates"        | [ 8,  8, 2, 2]   | [ 5, 4, 2, 2, 2] | same as cpc      | [ 5, 4, 4, 2, 2] |
| "upsample_kernel_sizes" | [16, 16, 4, 4]   | [11, 8, 4, 4, 4] | same as cpc      | [11, 8, 8, 4, 4] |
| "segment_size"          | 8192             | 8960             | same as cpc      | same as cpc      |
| "sampling_rate"         | 22050            | 16000            | same as cpc      | same as cpc      |
|                         |                  |                  |                  |                  |
| "model_in_dim"          | (80)             | 384              | same as cpc      | same as cpc      |
| "num_embeddings"        | -                | 100              | 256              | same as cpc      | 
| "code_hop_size"         | (256, ~86Hz)     | 160 (100Hz)      | same as cpc      | 320 (50Hz)       |
| others                  | -                | many new params  | same as cpc      | same as cpc      |


V1と比較して  

- segment長を10%延長
- srを22.05kから16kに変更
- 'up-MRF'レイヤーを1層追加

HuBERTはHopが伸びた（フレームレートが小さい）分、layer#3 を 2↑ -> 4↑ に変更。  
