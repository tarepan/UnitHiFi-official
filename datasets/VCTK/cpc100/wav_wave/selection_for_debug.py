from pathlib import Path

def generate_speaker_ristricted_data():
    for name in ["train", "val", "test"]:
        rel = "datasets/VCTK/cpc100/wav_wave"
        path_original = Path(f"{rel}/{name}.txt")
        path_target   = Path(f"{rel}/debug_{name}.txt")

        with open(path_target, mode="a") as f_target:
            with open(path_original) as f_original:
                for line in f_original.readlines():
                    if line[0] == '{':
                        # {"audio": "<path>", "SSL_type: "X X X ...", "duration": 1.9}
                        sample = eval(line.strip())
                        if "p304" in sample["audio"]:
                            f_target.write(line)


if __name__ == '__main__':
    generate_speaker_ristricted_data()
