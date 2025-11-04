import torch
import numpy as np
from snac import SNAC


class SNACCoder:
    def __init__(self, device_rank: int, model_id: str = "hubertsiuzdak/snac_24khz"):
        """
        Initialize SNAC codec
        
        Args:
            device_rank: CUDA device index
            model_id: SNAC model ID
                - "hubertsiuzdak/snac_24khz" (3 layers, 0.98 kbps, speech)
                - "hubertsiuzdak/snac_32khz" (4 layers, 1.9 kbps, music/SFX)
                - "hubertsiuzdak/snac_44khz" (4 layers, 2.6 kbps, music/SFX)
        """
        self.snac_model = SNAC.from_pretrained(model_id).eval()
        self.device = torch.device(f"cuda:{device_rank}")
        torch.cuda.set_device(self.device)
        self.snac_model = self.snac_model.to(self.device)
        self.model_id = model_id
        
        if "24khz" in model_id:
            self.num_layers = 3
        elif "32khz" in model_id or "44khz" in model_id:
            self.num_layers = 4
        else:
            self.num_layers = None

    def __call__(self, waveform: np.ndarray) -> dict:
        """
        Encode audio waveform to SNAC tokens
        
        Args:
            waveform: Audio waveform as numpy array (mono audio expected)
        
        Returns:
            Dictionary with encoded tokens at each layer and metadata
        """
        audio_tensor = torch.from_numpy(waveform).unsqueeze(dim=0)
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(1)
        audio_tensor = audio_tensor.to(dtype=torch.float32)
        audio_tensor = audio_tensor.to(self.device)
        
        with torch.inference_mode():
            codes = self.snac_model.encode(audio_tensor)
        
        if self.num_layers is None:
            self.num_layers = len(codes)
        
        encoded_audio = {}
        for i, code in enumerate(codes, start=1):
            encoded_audio[f'snac_layer_{i}'] = code.squeeze().cpu().numpy()
        
        encoded_audio['num_layers'] = self.num_layers
        encoded_audio['token_lengths'] = [code.shape[1] for code in codes]
        
        return encoded_audio
