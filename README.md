# Pathologies

This is an implementation of our paper: [Revisiting Pathologies of Neural Models under Input Reduction](https://aclanthology.org/2023.findings-acl.730/).
If you find our code and/or models useful, please cite our paper:

```bash
@inproceedings{kruengkrai-yamagishi-2023-revisiting,
    title = "Revisiting Pathologies of Neural Models under Input Reduction",
    author = "Canasai Kruengkrai and Junichi Yamagishi",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    year = "2023",
    url = "https://aclanthology.org/2023.findings-acl.730",
    pages = "11504--11517",
}
```

## Requirements

The code is tested on Python 3.9 and PyTorch 1.12.1.

We recommend to create a new environment for experiments using conda:
```shell
conda create -y -n pathologies python=3.9
conda activate pathologies

```

Then, clone the project and install necessary packages:
```shell
git clone https://github.com/nii-yamagishilab/pathologies
cd pathologies
pip install -r requirements.txt
```

For further development or modification, we recommend installing `pre-commit`:
```shell
pre-commit install
```

To ensure that PyTorch is installed and CUDA works properly, run:
```shell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

We should see:
```shell
1.12.1+cu113
True
```

:warning: We use PyTorch 1.12.1 with CUDA 11.3. You may need another CUDA version suitable for your environment.

## Experiments

See [experiments](experiments).

## Acknowledgments

This work is supported by JST CREST Grants (JPMJCR18A6 and JPMJCR20D3), JST AIP challenge program, and MEXT KAKENHI Grants (21H04906), Japan.

## Licence

BSD 3-Clause License

Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
