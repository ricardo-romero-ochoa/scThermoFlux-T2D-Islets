# Dataset notes and selection logic

## Primary dataset: GSE221156

Use as the primary discovery cohort because it is the largest currently identified public human islet T2D scRNA-seq dataset in this search. GEO describes it as a 10x Genomics single-cell RNA-seq atlas of 245,878 human islet cells from 48 cadaveric organ donors: 17 non-diabetic, 14 pre-diabetic, and 17 type 2 diabetic donors.

Recommended analysis:

- Start with beta cells.
- Compare ND versus T2D.
- Keep PD as an intermediate-state exploratory analysis.
- Aggregate thermodynamic observables by donor.
- Use within-cell-type analysis to avoid composition artifacts.

## Validation datasets

### GSE153855

Smart-seq2 data from 6 control and 5 T2D human donors. Useful for validation of module-level landscape deformation. It likely lacks spliced/unspliced layers, so use graph/pseudotime/disease-axis transition inference rather than RNA velocity.

### E-MTAB-5061

Segerstolpe/Palasantza dataset. EBI/HCA report 2,942 cells from 6 healthy and 4 T2D cadaveric donors. Good validation dataset with rich cell-type annotation, but expression scale and metadata harmonization need care.

### GSE81608

Xin et al. single-cell endocrine islet dataset with 1,492 human alpha, beta, delta and PP cells from non-diabetic and T2D organ donors. Useful endocrine-specific validation.

### GSE86469

Lawlor et al. dataset: 638 single cells from 5 ND and 3 T2D donors. Smaller but useful as independent replication.

### GSE83139

Mixed juvenile/adult/T1D/T2D donor classes. Use as cautious exploratory validation only; age and donor class can confound disease effects.

## Bulk anchor: GSE164416

Whole-islet RNA-seq from 133 donors: 18 ND, 41 IGT, 35 T3cD, and 39 T2D. This should not be used for single-cell flux but can validate module behavior and connect with prior ML/meta-analysis work.

## Practical recommendation

Manuscript v1 should use:

1. Synthetic validation.
2. GSE221156 beta-cell ND vs T2D discovery.
3. GSE153855 or E-MTAB-5061 validation.
4. GSE164416 bulk module anchor as supplementary validation only.
