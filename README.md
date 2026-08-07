The softmax normalization in transformer attention
is seen as a bottleneck for custom accelerators since each row
involves an exponential computation, a full-row summation, and
a division, none of which is hardware-friendly. We propose
a fixed-point (Q8.8) softmax accelerator evaluated across two
design spaces. Architecturally, we propose six datapath variants,
spanning distinct exponential-unit and pipeline-depth choices,
tailored to maximize hardware benefits in the form of area,
frequency, and throughput. The deep-pipelining method raises
operating frequency by up to 12.5× (M1, flat vs. deep-pipelined)
over a flat datapath at a comparable LUT cost. On this unit, we
add Safe-Gated Dynamic Data-dependent Filtering (SG-DDF), a
one-comparator gate that prunes 32-41% of tokens when tested
on GPT-2 Medium and Qwen 2.5 on peaked attention rows. While
provably reverting to exact stable softmax on flat rows, generating
sparsity at no accuracy cost,it gives 9.4× the operating frequency
for the deployed SG-DDF(M3) configuration. The accelerator is
validated on FPGA and ASIC flows. The same bit-exact hardware
model is further evaluated as a drop-in softmax replacement,
without any retraining, on two language models (GPT-2 Medium
and Qwen2.5, across WikiText-103 and C4) and on ImageNet-
pretrained CNN classifiers (ResNet50 and DenseNet121, across
CIFAR-10, CIFAR-100). Across every configuration investigated,
Stable and SG-DDF add at most 2.0 perplexity to the exact-
softmax baseline and move top-1 accuracy by no more than 0.1
points, with SG-DDF remaining equivalent to baseline softmax
throughout. The unstabilized (M0) mode, by contrast, fails as
predicted wherever the maximum-subtraction precondition is
skipped, overflowing the reciprocal on up to 52% of rows and,
in the worst case, driving perplexity from a baseline of∼20.5
(Qwen2.5) past 2 × 105 or collapsing CNN top-1 accuracy by
up to 74.1 points. 
