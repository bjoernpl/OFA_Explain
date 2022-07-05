from OfaExplanationGenerator import GeneratorOurs
from explanation_generator import ExplanationGenerator, apply_half
from PIL import Image
import torch

from fairseq import utils, tasks

def run():
    ex_gen = ExplanationGenerator()
    generator = GeneratorOurs(ex_gen.models[0], ex_gen.task, ex_gen.generator)
    image = Image.open("./examples/showcase1.jpg")
    sample = ex_gen.construct_sample(image, "Describe the image")
    sample = utils.move_to_cuda(sample)
    sample = utils.apply_to_sample(apply_half, sample)
    generator.generate_ours(sample)

if __name__ == "__main__":
    run()