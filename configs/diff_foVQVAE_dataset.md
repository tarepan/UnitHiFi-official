## Config Difference in foVQVAE

| parameter name          | LJSpeech                             | VCTK                                               |
| ----------------------- | ------------------------------------ | -------------------------------------------------- |
| "input_training_file"   | "datasets/LJSpeech/cpc100/train.txt" | "datasets/VCTK/vctk_audio_text_train_filelist.txt" |
| "input_validation_file" | "datasets/LJSpeech/cpc100/val.txt"   | "datasets/VCTK/vctk_audio_text_val_filelist.txt"   |
| "f0_stats"              | "datasets/LJSpeech/f0_stats.pth"     | "datasets/VCTK/f0_stats.th"                        |
| "multispkr"             | "single"                             | "multispkr"                                        |

- waveとfoのパスが異なる
- speakerのsingle/multi
