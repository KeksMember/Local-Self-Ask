# Local-Self-Ask
A self-ask clone running locally without any paid services

Credits:
The self-ask paper: https://arxiv.org/pdf/2210.03350.pdf
The llama.cpp Python bindings used in the code: https://github.com/abetlen/llama-cpp-python
The search engine used: https://github.com/searx/searx

Description:

To use this code please install the above mentioned python bindings for llama.cpp, download a suitable ggml model which you also specify in the model_path inside the Llama initialization. Next set up a SearX instance inside a Docker container or wherever you would like to and replace the URL inside the code with yours. And you're as good as done, change the model parameters to your liking(Don't forget to set the threads count to your CPU's core count so the model works at peak speeds) and enter your question into the run methods parameter!
