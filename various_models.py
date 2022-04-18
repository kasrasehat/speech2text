import fairseq
import torch
from fairseq.models import wav2vec

cp = torch.load("wav2vec_small_960h.pt")
model = wav2vec.base_wav2vec_architecture(cp['args'])
model.load_state_dict(cp['model'])
model.eval()
#cp = torch.load("wav2vec_small_960h.pt")
#model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(["wav2vec_small_960h.pt"], arg_overrides={"data": "E:/codes_py/speech2text/data"})
#model = model[0]
#model.eval()

wav_input_16khz = torch.randn(1,10000)
z = model.feature_extractor(wav_input_16khz)
_, idxs = model.vector_quantizer.forward_idx(z)
print(idxs.shape) # output: torch.Size([1, 60, 2]), 60 timesteps with 2 indexes corresponding to 2 groups in the model