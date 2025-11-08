# Training Commands

## Quick Validation

```bash
python -m mace.cli.run_train  --name="rmd17-Aspirin"  --train_file="repro_data/dataset/dataset_rmd17/revised_aspirin/train.xyz"  --valid_fraction=0.5  --test_file="repro_data/dataset/dataset_rmd17/revised_aspirin/test.xyz"  --E0s="average"  --model="MACE"  --loss="ef"  --num_interactions=2  --num_channels=128  --max_L=1  --correlation=3  --r_max=5.0  --lr=0.005  --forces_weight=1000  --energy_weight=1  --weight_decay=1e-8  --clip_grad=100  --batch_size=5  --valid_batch_size=5  --max_num_epochs=500  --scheduler_patience=5  --ema  --ema_decay=0.995  --swa  --start_swa=400  --error_table="TotalMAE"  --default_dtype="float64" --device=cuda  --seed=123  --save_cpu  --energy_key="REF_energy"  --forces_key="REF_forces"  --foundation_model="repro_data/foundation_model/MACE-OFF23_medium.model" 
```

## Main experiments

### For **rMD17** Dataset

```bash
python -m mace.cli.run_train  --name="rmd17-[MOLECULE]"  --train_file="repro_data/dataset/dataset_rmd17/revised_[MOLECULE]/train.xyz"  --valid_fraction=0.5  --test_file="repro_data/dataset/dataset_rmd17/revised_[MOLECULE]/test.xyz"  --E0s="average"  --model="MACE"  --loss="ef"  --num_interactions=2  --num_channels=128  --max_L=1  --correlation=3  --r_max=5.0  --lr=0.005  --forces_weight=1000  --energy_weight=1  --weight_decay=1e-8  --clip_grad=100  --batch_size=5  --valid_batch_size=5  --max_num_epochs=500  --scheduler_patience=5  --ema  --ema_decay=0.995  --swa  --start_swa=400  --error_table="TotalMAE"  --default_dtype="float64" --device=cuda  --seed=123  --save_cpu  --energy_key="REF_energy"  --forces_key="REF_forces"  --foundation_model="repro_data/foundation_model/MACE-OFF23_medium.model" 
```

Replace `[MOLECULE]` according to the actual dataset name.



---

### For **AcAc** Dataset


```bash
python -m mace.cli.run_train  --name="AcAc-[TEMP]"  --train_file="repro_data/dataset/dataset_acac/train_300K.xyz"  --valid_fraction=0.1  --test_file="repro_data/dataset/dataset_acac/test_MD_[TEMP].xyz"  --E0s="average"  --model="MACE"  --loss="ef"  --num_interactions=2  --num_channels=128  --max_L=1  --correlation=3  --r_max=5.0  --lr=0.005  --forces_weight=1000  --energy_weight=1  --weight_decay=1e-8  --clip_grad=100  --batch_size=5  --valid_batch_size=5  --max_num_epochs=500  --scheduler_patience=5  --ema  --ema_decay=0.995  --error_table="TotalRMSE"  --default_dtype="float64" --device=cuda  --seed=123  --save_cpu  --energy_key="energy"  --forces_key="forces"   --foundation_model="repro_data/foundation_model/MACE-OFF23_medium.model" 
```

Replace `[TEMP]` with `300K` or `600K` according to the actual dataset name.


---

### For **3BPA** Dataset

```bash
MMEA_RANK=32 MMEA_STD=0.001 python -m mace.cli.run_train  --name="3BPA-[TEMP]"  --train_file="repro_data/dataset/dataset_3BPA/train_300K.xyz"  --valid_fraction=0.1  --test_file="repro_data/dataset/dataset_3BPA/test_[TEMP].xyz"  --E0s="average"  --model="MACE"  --loss="ef"  --num_interactions=2  --num_channels=128  --max_L=1  --correlation=3  --r_max=5.0  --lr=0.005  --forces_weight=1000  --energy_weight=1  --weight_decay=1e-8  --clip_grad=100  --batch_size=5  --valid_batch_size=5  --max_num_epochs=500  --scheduler_patience=5  --ema  --ema_decay=0.995  --swa  --start_swa=400  --error_table="TotalRMSE"  --default_dtype="float64" --device=cuda  --seed=123  --save_cpu  --energy_key="energy"  --forces_key="forces"   --foundation_model="repro_data/foundation_model/MACE-OFF23_medium.model"
```

Replace `[TEMP]` with `300K`, `600K`, `1200K`, or `dih` according to the actual dataset name.


## Evaluation

Evaluation is performed using `RMSE` and `MAE`. Note that for some datasets, SWA can be used during training. 