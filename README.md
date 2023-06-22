<div align=center>
  <h1>Control Oriented Octopus Muscle Model (COOMM)</h1>
</div>

## Project Description

Control Oriented Muscle Model for an octopus arm. The package extend the implementation of Cosserat Rod simulation done by [PyElastica](https://github.com/GazzolaLab/PyElastica).

## Install

The package can be installed using standard python package installer (pip):

```bash
pip install coomm
```

or from source code:

```bash
cd <repository directory>
git clone https://github.com/hanson-hschang/COOMM
pip install -e COOMM
```

## Examples

> Note, all the example cases written uses the original version of COOMM (v.0.0.1), with older version of PyElastica (v.0.2.2). If you are interested in reproducing the result from the paper, please match the version.

## Citation

We ask that any publications which use COOMM cite as following:

```
@article{chang2022energy,
  title={Energy Shaping Control of a Muscular Octopus Arm Moving in Three Dimensions},
  author={Chang, Heng-Sheng and Halder, Udit and Shih, Chia-Hsien and Naughton, Noel and Gazzola, Mattia and Mehta, Prashant G},
  journal={arXiv preprint arXiv:2209.04089},
  year={2022}
}
```

> Original implementation for the paper is in [version 0.0.1](https://github.com/hanson-hschang/COOMM/tree/v0.0.1.post2).

## References

- Gazzola, Dudte, McCormick, Mahadevan, <strong>Forward and inverse problems in the mechanics of soft filaments</strong>, Royal Society Open Science, 2018. doi: [10.1098/rsos.171628](https://doi.org/10.1098/rsos.171628)

## List of publications and submissions

- [A physics-informed , vision-based method to reconstruct all deformation modes in slender bodies](https://arxiv.org/abs/2109.08372) (UIUC 2021) (IEEE ICRA 2022) [code]( https://github.com/GazzolaLab/BR2-vision-based-smoothing)

## Related Works

- PyElastica: https://github.com/GazzolaLab/PyElastica
