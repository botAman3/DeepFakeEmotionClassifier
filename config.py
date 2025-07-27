import torch 

def get_config():
    config = {
        "data_dir_original" : "path to original audios ",
        "data_dirs_fake": [
            'path to dac codec deep fake audio',
            'path to encodec codec deep fake audio',
            'path to speechtokenizer codec deep fake audio',
        ],

        "model_save_path" = "Add your path",

        "target_sample_rate" : 16000,
        "max_duration_seconds" : 10 ,
        "n_fft" : 1024,
        "hop_length" : 512 , 
        "n_mels" : 128 , 
        "vit_image_size":224 

        "num_epochs": 10,
        "batch_size": 16,
        "learning_rate": 1e-5,
        "test_split_size": 0.2,
        "grad_clip_norm": 1.0,
        "random_state": 42,


        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 4

    }


    return config
    