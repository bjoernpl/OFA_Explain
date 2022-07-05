from OfaExplanationGenerator import GeneratorOurs
from explanation_generator import ExplanationGenerator, apply_half
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from fairseq import utils, tasks

def run():
    ex_gen = ExplanationGenerator()
    generator = GeneratorOurs(ex_gen.models[0], ex_gen.task, ex_gen.generator)
    image = Image.open("./Darwin_OP_1.jpg")
    sample = ex_gen.construct_sample(image, "Does the robot have ears?")
    sample = utils.move_to_cuda(sample)
    # sample = utils.apply_to_sample(apply_half, sample)
    self_attn_map, result = generator.generate_ours(sample)
    answer = result[0]["answer"]
    print(answer)
    num_tokens = int(generator.text_tokens)
    num_patches = int(generator.image_patches)
    fig, axs = plt.subplots(1, num_tokens, figsize=(5 * num_tokens, 5))
    tokens = sample["net_input"]["src_tokens"].squeeze().cpu().numpy()
    print(tokens)
    for i, token in enumerate(tokens):

        attn_map = self_attn_map.squeeze(0).detach().cpu()
        text_attn = attn_map.T[num_patches + i, -num_tokens+1:-1]
        text_attn -= text_attn.min()
        text_attn /= text_attn.max()
        print(text_attn)
        image_attn = attn_map.T[num_patches + i, :-num_tokens]
        heatmap = torch.reshape(image_attn, (1, 1, 24, 24))
        heatmap_img = F.interpolate(heatmap, image.size[::-1], mode='bicubic')
        heatmap_img = heatmap_img.squeeze(0).squeeze(0).numpy()

        axs[i].imshow(image)
        axs[i].imshow(heatmap_img, alpha=0.7)
        axs[i].axis('off')
        axs[i].set_title(token)
    plt.savefig("./encoder_attention.pdf")
    plt.show()
    plt.close()


if __name__ == "__main__":
    run()