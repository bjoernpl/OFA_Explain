### What is this demo?
This is a demonstration of the [OFA](https://github.com/ofa-sys/ofa) visual question answering model with modifications
made to implement visual explanations of the underlying attention layers. The method for generating
these explanations is based on the work by [Chefer et al.](https://github.com/hila-chefer/Transformer-MM-Explainability)
.

### What is OFA?

OFA (One for all) is a unified multimodal pre-trained model that unifies modalities
(i.e., cross-modality, vision, language) and tasks (e.g., image generation, visual grounding,
image captioning, image classification, text generation, etc.) to a simple sequence-to-sequence
learning framework. It was created by the Alibaba Group and the associated DAMO academy. For further information see the
ICML 2022 paper by Wang et al.
[OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](http://arxiv.org/abs/2202.03052)
.
In our case, we are using the large model pretrained on a Visual Question Answering (VQA) task.

### How does the demo work?

To use this demo, simply select one of the given images (or upload your own) and pose a question
in the text field below. When you click process, the image is processed by the OFA model, and our explanations are
added. In the right column, you will see the model's answer. Click on any of the words
in the answer to see the associated attention map on your image.

### Run locally?
#### Frontend
You can run this website as well as the inference API locally. The frontend is built with React
and hosted via Cloudflare. To run, clone [the repository](https://github.com/lukasbraach/explainable-ofa)
and run ``npm install`` followed by ``npm start``.

#### Backend
The inference API is available in [this repository](https://github.com/bjoernpl/OFA_Explain). OFA is implemented
with PyTorch and the API is built with [FastAPI](https://fastapi.tiangolo.com/) and run via [Uvicorn](https://www.uvicorn.org/). 

To install simply clone the repo, run 
``
pip install -r requirements.txt
``
to install all requirements, and
start the uvicorn server with 
``
python -m uvicorn api:app --reload --host 0.0.0.0
``

### Contact
This project was done as part of a seminar at University of Hamburg by
[Lukas Braach](https://github.com/lukasbraach) and [Björn Plüster](https://github.com/bjoernpl).
For any questions, feel free to contact us. Enjoy!
