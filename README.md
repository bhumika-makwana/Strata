## Strata
Mechanistically-informed model structures for transcriptomics perturbation effect prediction. 

Strata is a research framework for predicting transcriptional responses to genetic perturbations using single-cell RNA-seq data and curated molecular interaction priors.

#### Hypothesis
We hypothesize that gene-level perturbation responses are coupled through the underlying regulatory and signaling topology of the cell, rather than independent. Strata encodes this coupling structurally: a learnable gene interaction matrix modulates pairwise attention between gene representations, constraining the transformer to route information along biologically plausible interaction paths.

Instead of treating a biological network as a fixed ground-truth graph which is the case with current state of the art approaches, Strata initializes a learnable graph from curated prior knowledge and allows perturbational data to refine the strength of these relationships during training.

