# MVCL-DTI adaptation notice

`mvcl_cti_model.py` adapts mechanisms from https://github.com/ljquanlab/MVCL-DTI
at commit `9851641f1c82d12d0e35aa2403505e3f51f92651` (JCIM 2025,
DOI 10.1021/acs.jcim.4c02073). It is not an author reproduction.
Changes include four typed relation-only inputs, fixed random features, explicit
node ordering, guarded small-graph execution, PyTorch propagation without DGL,
and four specified metapaths. See docs/MVCL_DTI_IMPLEMENTATION_AUDIT.md.

Upstream license:

MIT License

Copyright (c) 2022 Dandelion

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
