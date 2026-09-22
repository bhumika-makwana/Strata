# Strata
Mechanistically-informed model structures for transcriptomics perturbation effect prediction. 

Strata is a research framework for predicting transcriptional responses to genetic perturbations using single-cell RNA-seq data and curated molecular interaction priors.

The central hypothesis is that perturbation responses are not independent across genes. Regulatory and signaling relationships define plausible routes through which a perturbation can propagate through a cellular system. Strata incorporates this prior structure into a transformer-based model by using a learnable gene interaction matrix to modulate attention between gene representations.

Instead of treating a biological network as a fixed ground-truth graph which is the case with current state of the art approaches, Strata initializes a learnable graph from curated prior knowledge and allows perturbational data to refine the strength of these relationships during training.

