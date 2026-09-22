## Strata
Mechanistically-informed model structures for transcriptomics perturbation effect prediction. 

Strata is a research framework for predicting transcriptional responses to genetic perturbations using single-cell RNA-seq data and curated molecular interaction priors.

**Core contribution** - A learnable, biologically guided perturbation-propagation model that uses curated gene regulatory/signaling networks as a prior to model genetic perturbations across the transcriptome and generalize to unseen gene combinations. Instead of treating a biological network as a fixed ground-truth graph which is the case with current state of the art approaches, Strata initializes a learnable graph from curated prior knowledge and allows perturbational data to refine the strength of these relationships during training.

**Prior-knowledge datasets** - 
We use the Norman et al. (2019) Perturb-seq dataset as the primary experimental dataset, containing single-cell transcriptomic profiles of K562 cells under genetic perturbations. To incorporate prior biological knowledge, we construct a gene interaction network by integrating CollecTRI, a curated transcription factor–target regulatory resource, with OmniPath, a curated resource of molecular and causal signaling interactions.

| Resource        | Description                                             | Role                                         |
| --------------- | ------------------------------------------------------- | -------------------------------------------- |
| **CollecTRI**   | TF–target gene regulatory network                       | **Transcriptional regulation**               |
| **OmniPath**    | Curated molecular and signaling interaction network     | **Signaling and protein-level interactions** |
| **Norman 2019** | Single-cell gene expression under genetic perturbations | **Perturbation-response data**               |


**Architecture** - 

                         ┌──────────────────────┐
                         │      G_prior         │
                         │ CollecTRI + OmniPath │
                         └──────────┬───────────┘
                                    │
                              initialization
                                    ▼
                         ┌──────────────────────┐
                         │  Learnable Gene      │
                         │  Interaction Graph   │
                         │        Gθ            │
                         └──────────┬───────────┘
                                    │
                                    │ guide
                                    ▼
┌────────────────┐       ┌──────────────────────┐
│ Control        │       │ Graph-Guided         │
│ expression     ├──────►│ Transformer          │
│                │       │                      │
│ Perturbations  ├──────►│ Gene tokens          │
└────────────────┘       │ Attention × 2        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Predicted            │
                         │ post-perturbation    │
                         │ expression           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                              ┌───────────┐
                              │   Loss    │
                              │ MSE +     │
                              │ Graph     │
                              │ regular.  │
                              └─────┬─────┘
                                    │
                              backpropagation
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
                 ▼                                     ▼
        Update model parameters              Update Gθ
                 │                                     │
                 └──────────────────┬──────────────────┘
                                    │
                                    │ repeat every
                                    │ training step
                                    └───────────────►
