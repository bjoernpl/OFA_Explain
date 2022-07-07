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
    image = Image.open("../frontend-ofa/public/images/horse_small.jpg")
    question = "Does the horse have ears ?"
    sample = ex_gen.construct_sample(image, question)
    sample = utils.move_to_cuda(sample)
    # sample = utils.apply_to_sample(apply_half, sample)
    self_attn_map, result = generator.generate_ours(sample)
    answer = result[0]["answer"]
    print(answer)
    num_tokens = int(generator.text_tokens)-2
    num_patches = int(generator.image_patches)
    fig, axs = plt.subplots(1, num_tokens + 2, figsize=(5 * (num_tokens + 2), 5))
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[0].set_title("Input image")

    tokens = sample["net_input"]["src_tokens"].squeeze().cpu().numpy()
    print(tokens)
    result_attn = result[0]["attention"]
    for i, attn_map in enumerate(result_attn.T.cpu()[:-1]):
        num_input_tokens = len(attn_map) - 576
        image_attn = attn_map[:-num_input_tokens]
        self_attn = attn_map[-num_input_tokens:]
        print("self attn", self_attn)
        image_attn = F.softmax(image_attn)
        heatmap = torch.reshape(image_attn, (1, 1, 24, 24))
        heatmap_img = F.interpolate(heatmap, image.size[::-1], mode='bicubic')
        heatmap_img = heatmap_img.squeeze(0).squeeze(0).numpy()
        axs[-1].imshow(image)
        axs[-1].imshow(heatmap_img, alpha=0.7)
        axs[-1].axis('off')
        axs[-1].set_title(answer.split()[0])

    for i, token in enumerate(tokens[1:-1]):
        attn_map = self_attn_map.squeeze(0).detach().cpu()
        text_attn = attn_map.T[num_patches + i, -num_tokens:-1]
        text_attn -= text_attn.min()
        text_attn /= text_attn.max()
        print(text_attn)
        image_attn = attn_map.T[num_patches + i + 1, :-num_tokens-2]
        heatmap = torch.reshape(image_attn, (1, 1, 24, 24))
        heatmap_img = F.interpolate(heatmap, image.size[::-1], mode='bicubic')
        heatmap_img = heatmap_img.squeeze(0).squeeze(0).numpy()

        axs[i + 1].imshow(image)
        axs[i + 1].imshow(heatmap_img, alpha=0.7)
        axs[i + 1].axis('off')
        axs[i + 1].set_title(question.split()[i])
    plt.tight_layout()
    plt.savefig("./output_plot.pdf")
    plt.show()
    plt.close()


if __name__ == "__main__":
    run()