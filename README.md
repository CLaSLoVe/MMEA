# MMEA: *Magnitude‑Modulated Equivariant Adapter for Parameter‑Efficient Fine‑Tuning of Equivariant GNNs*

`MMEA` introduces a lightweight adapter that scales each multiplicity block by scale gating, achieving SOTA performance in PEFT of SO(3)-equivariant GNNs for molecular force field prediction.

<div align="center">
  <img src="mmea.png" alt="Magnitude-Modulated Equivariant Adapter" style="width:300px; max-width:100%;" />
</div>

---

## Installation

First install pytorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 # follow your env
```


Install all required packages with the following command:

```bash
pip install ase opt_einsum opt_einsum_fx icecream torch_ema torchmetrics matscipy h5py prettytable hostlist
```
---

## Quick Validation

You can run the following for a quick test on `rMD17-Aspirin`:


```bash
python -m mace.cli.run_train  --name="rmd17-Aspirin"  --train_file="repro_data/dataset/dataset_rmd17/revised_aspirin/train.xyz"  --valid_fraction=0.5  --test_file="repro_data/dataset/dataset_rmd17/revised_aspirin/test.xyz"  --E0s="average"  --model="MACE"  --loss="ef"  --num_interactions=2  --num_channels=128  --max_L=1  --correlation=3  --r_max=5.0  --lr=0.005  --forces_weight=1000  --energy_weight=1  --weight_decay=1e-8  --clip_grad=100  --batch_size=5  --valid_batch_size=5  --max_num_epochs=500  --scheduler_patience=5  --ema  --ema_decay=0.995  --swa  --start_swa=400  --error_table="TotalMAE"  --default_dtype="float64" --device=cuda  --seed=123  --save_cpu  --energy_key="REF_energy"  --forces_key="REF_forces"  --foundation_model="repro_data/foundation_model/MACE-OFF23_medium.model" 
```
---

## Datasets

Our datasets include `rMD17`, `3BPA`, and `AcAc`, which are the same as those used in `MACE`.

Among them, `3BPA` and `AcAc` are used for comprehensive evaluation, while `rMD17`, being relatively simple, is used to assess the model's few-shot learning capabilities. For fairness, we adopt the same data splits for training.

To facilitate reproducibility, we provide the final data splits for `rMD17` in `repro_data/dataset/dataset_rmd17`. The original `rMD17` dataset can be downloaded from [2], and the `3BPA` and `AcAc` datasets are available at BOTNet [3].

---

## Baselines

Full-parameter fine-tuning and `ELoRA` fine-tuning use the setup described in the `ELoRA` repository.

---

## Training with `MMEA`

With the environment active, any standard `mace.cli.run_train` command will automatically load the `MMEA` adapter.
See `repro_data/run.md` for more detailed information.
Example:

```bash
python -m mace.cli.run_train \
  --name "[EXPERIMENT_NAME]" \
  --train_file "[PATH/TO/train.xyz]" \
  --valid_fraction [FRACTION] \
  --test_file "[PATH/TO/test.xyz]" \
  --foundation_model "[PATH/TO/pretrained.model]" \
  --model "MACE" \
  --loss "ef" \
  --num_interactions 2 \
  --num_channels 128 \
  --max_L 1 \
  --r_max 5.0 \
  --lr 0.005 \
  --forces_weight 1000 \
  --energy_weight 1 \
  --batch_size 5 \
  --max_num_epochs [EPOCHS] \
  --ema \
  --ema_decay 0.995 \
  --error_table "[TotalMAE/TotalRMSE]" \
  --device cuda \
  --seed [SEED] \
  --save_cpu
  # ---- optional: SWA ----
  --swa \
  --start_swa [SWA_EP]
```

**Important Note**: During training, you should see the following message:

```
====================================================================================================
 MMEA_RANK:  16
====================================================================================================
```
This indicates that the program is running correctly.

---

## Acknowledgements

We would like to express our sincere gratitude to:

* **ELoRA** [4] for releasing their low-rank adaptation baseline, and
* **MACE** [1] for providing the foundation on which both `ELoRA` and `MMEA` are built.

We are grateful for their excellent work and open-source contributions to the community's subsequent research.


## References

[1] Batatia I, Kovacs D P, Simm G, et al. MACE: Higher order equivariant message passing neural networks for fast and accurate force fields[J]. Advances in neural information processing systems, 2022, 35: 11423-11436.

[2] Christensen A S, Von Lilienfeld O A. On the role of gradients for machine learning of molecular energies and forces[J]. Machine Learning: Science and Technology, 2020, 1(4): 045018.

[3] Batatia I, Batzner S, Kovács D P, et al. The design space of E (3)-equivariant atom-centred interatomic potentials[J]. Nature Machine Intelligence, 2025, 7(1): 56-67.

[4] Wang C, Hu S, Tan G, et al. ELoRA: Low-Rank Adaptation for Equivariant GNNs[C]//Forty-second International Conference on Machine Learning.